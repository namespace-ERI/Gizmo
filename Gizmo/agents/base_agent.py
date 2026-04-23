import copy
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI

"""
    对所有Agent的基类，提供了LLM调用、工具调用、解析响应、执行工具等基本功能。不同厂商的template不同所以为了适配不同模型，需要重写一些方法。
"""


@dataclass
class LLMConfig:
    """LLM 调用的生成参数配置。

    Attributes:
        max_tokens:       最大输出 token 数（chat.completions 兼容字段；GPTAgent
                          会回退为 Responses API 的 max_output_tokens）。
        max_output_tokens: Responses API 最大输出 token 数。
        temperature:      采样温度。
        top_p:            nucleus sampling 参数。
        seed:             随机种子，用于复现。
        timeout:          HTTP 请求超时（秒），同时作为 OpenAI 客户端超时。
        stream:           是否使用 chat.completions 流式返回；None 表示由具体
                          Agent 决定默认行为。
        enable_thinking:  是否开启 vllm thinking 模式（通过 extra_body 注入
                          chat_template_kwargs.enable_thinking=True）。
        extra_body:       透传给 OpenAI API 的额外请求体字段（如 vllm 扩展参数），
                          与 enable_thinking 产生的字段深度合并。
        store:            Responses API store 字段。
        truncation:       Responses API truncation 字段。
        parallel_tool_calls: Responses API parallel_tool_calls 字段。
        tool_choice:      Responses API tool_choice 字段。
        reasoning:        Responses API reasoning 对象。
        reasoning_effort: Responses API reasoning.effort 字段。
        reasoning_summary: Responses API reasoning.summary 字段。
        text:             Responses API text 对象。
        text_verbosity:   Responses API text.verbosity 字段。
        text_format:      Responses API text.format 字段。
        include:          Responses API include 字段。
        metadata:         Responses API metadata 字段。
        service_tier:     Responses API service_tier 字段。
        prompt_cache_key: Responses API prompt_cache_key 字段。
        safety_identifier: Responses API safety_identifier 字段。
    """
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    timeout: float = 120.0
    stream: Optional[bool] = None
    enable_thinking: bool = False
    extra_body: Optional[dict] = None
    store: Optional[bool] = None
    truncation: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    tool_choice: Optional[object] = None
    reasoning: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text: Optional[dict] = None
    text_verbosity: Optional[str] = None
    text_format: Optional[dict] = None
    include: Optional[list[str]] = None
    metadata: Optional[dict] = None
    service_tier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    safety_identifier: Optional[str] = None


@dataclass
class RunState:
    """单次 run() 的运行状态，每次调用 run/run_verbose 时重置。

    Attributes:
        step:         当前已完成的 LLM 调用轮数（含最终回答轮）。
        tool_rounds:  调用了工具的轮数。
        elapsed:      距本次 run 开始经过的秒数（实时更新）。
        stop_reason:  停止原因："" 表示正常结束，否则为 "max_steps" /
                      "timeout" / "max_tool_rounds"。
    """
    step: int = 0
    tool_rounds: int = 0
    start_time: float = field(default_factory=time.time)
    stop_reason: str = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class ContextManager(ABC):
    """可插拔的消息上下文处理器基类。

    在每次 LLM 调用前，按注册顺序依次处理完整的 messages 列表（含 system 消息），
    可用于实现以下功能（接口预留，具体逻辑由子类实现）：
        - token budget:       统计 token 数，超出预算时截断或压缩历史。
        - history trim:       按轮数或时间窗口剪裁旧消息。
        - message rebuild:    重新格式化消息结构（如合并多轮 tool 消息）。
        - truncation notice:  在截断位置注入占位消息，告知模型历史已被截断。

    与 Hook 的区别：
        Hook (before_llm/after_llm/...)  是轻量级事件回调，observe/react，无返回值。
        ContextManager                   是消息变换管道，接收完整 messages 并返回变换结果。

    执行顺序（每次 LLM 调用前）：
        1. _apply_context_managers(messages)  → 依次经过所有 ContextManager
        2. _fire("before_llm", state, ...)    → 触发 before_llm hook（看到处理后的消息）
        3. client.chat.completions.create()   → 实际 LLM 调用
    """

    @abstractmethod
    def process(self, messages: list[dict], state: "RunState") -> list[dict]:
        """对完整 messages 列表进行变换，返回处理后的版本。

        Args:
            messages: 当前完整消息列表，第一条为 system 消息，其余按时序排列。
            state:    当前 RunState，可读取 step / tool_rounds / elapsed 等信息。

        Returns:
        变换后的 messages 列表，将直接传递给下一个 ContextManager 或 LLM。
            若无需修改，直接返回原列表即可。返回结果会在真正发起 LLM 请求前
            同步回写到 agent 的内部 messages 状态中。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """每次 run() 开始时调用，用于重置有状态的 ContextManager。

        默认无操作；有内部状态（如 token 计数器、已处理轮数）的子类应覆盖此方法。
        """


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: str


@dataclass
class TrajectoryStep:
    step: int
    reasoning: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_content: str = ""


class BaseAgent:
    _PARSE_RETRY_LIMIT = 1
    _FINALIZE_FOLLOWUP_LIMIT = 4

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[list] = None,
        max_steps: int = 200,
        max_time_seconds: Optional[float] = None,
        max_tool_rounds: Optional[int] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict] = []
        self.trajectory: list[TrajectoryStep] = []
        self.state: RunState = RunState()
        self.llm_config = llm_config or LLMConfig()

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.llm_config.timeout,
        )

        # Hook 注册表，每个事件对应一组有序的回调函数
        self._hooks: dict[str, list[Callable]] = {
            "before_llm":   [],   # (state, messages) -> None
            "after_llm":    [],   # (state, parsed)   -> None
            "after_tool":   [],   # (state, tool_name, args, result) -> None
            "should_stop":  [],   # (state)            -> Optional[str]
        }

        # ContextManager 链，按注册顺序依次处理消息
        self._context_managers: list[ContextManager] = []

    # ------------------------------------------------------------------
    # Hook 注册 API
    # ------------------------------------------------------------------

    def on(self, event: str, fn: Callable) -> "BaseAgent":
        """注册生命周期 hook，返回 self 支持链式调用。

        事件说明：
            before_llm(state, messages)           每次 LLM 调用前触发，可就地修改 messages。
            after_llm(state, parsed)              每次 LLM 返回后触发，parsed 含 final_content /
                                                  reasoning_content / tool_calls 等字段。
            after_tool(state, name, args, result) 每次工具执行完毕后触发。
            should_stop(state) -> Optional[str]   每轮循环开始时触发，返回非空字符串即触发停止；
                                                  多个 hook 中第一个非空返回优先。

        示例：
            agent.on("after_llm", lambda s, p: print(p["reasoning_content"]))
            agent.on("should_stop", lambda s: "custom" if s.step > 5 else None)
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event!r}. "
                             f"Valid events: {list(self._hooks)}")
        self._hooks[event].append(fn)
        return self

    def _fire(self, event: str, *args):
        """触发某事件的所有 hook，收集 should_stop 的返回值。"""
        for fn in self._hooks[event]:
            result = fn(*args)
            if event == "should_stop" and result:
                return result
        return None

    # ------------------------------------------------------------------
    # ContextManager 注册 API
    # ------------------------------------------------------------------

    def use(self, cm: ContextManager) -> "BaseAgent":
        """注册一个 ContextManager，返回 self 支持链式调用。

        ContextManager 按注册顺序形成管道，每次 LLM 调用前依次处理消息列表。
        消息列表包含 system 消息（第一条），随后是完整对话历史。

        示例：
            agent.use(TokenBudgetManager(max_tokens=4096))
                 .use(TruncationNoticeManager(notice="[历史已截断]"))
        """
        self._context_managers.append(cm)
        return self

    def _apply_context_managers(self, messages: list[dict]) -> list[dict]:
        """将 messages 依次传过所有已注册的 ContextManager，返回最终结果。"""
        for cm in self._context_managers:
            messages = cm.process(messages, self.state)
        return messages

    def _persist_processed_messages(self, messages: list[dict]) -> None:
        """将 ContextManager 处理后的完整消息列表同步回内部状态。"""
        if not messages:
            self.messages = []
            return

        if messages[0].get("role") == "system":
            self.messages = list(messages[1:])
            return

        self.messages = list(messages)

    def _use_native_tools(self) -> bool:
        return False

    def _build_extra_body(self) -> Optional[dict]:
        """合并 enable_thinking 标志和用户自定义 extra_body。"""
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        if cfg.enable_thinking:
            tmpl = base.setdefault("chat_template_kwargs", {})
            tmpl["enable_thinking"] = True

        return base or None

    def _default_stream(self) -> bool:
        return False

    def _should_stream_response(self) -> bool:
        if self.llm_config.stream is not None:
            return bool(self.llm_config.stream)
        return self._default_stream()

    def _prepare_request_kwargs(self, kwargs: dict) -> dict:
        """子类可在发送请求前调整 OpenAI-compatible 请求参数。"""
        return kwargs

    def _prepare_messages_for_llm(self, messages: list[dict]) -> list[dict]:
        """子类可在真正发起请求前校验或规范化消息列表。"""
        return messages

    def _call_llm(self):
        raw_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages = self._apply_context_managers(raw_messages)
        messages = self._prepare_messages_for_llm(messages)
        self._persist_processed_messages(messages)
        cfg = self.llm_config

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }

        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed

        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        if self._use_native_tools():
            tools = [tool.to_schema() for tool in self.tools.values()] or None
            if tools:
                kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: tool '{tool_name}' not found"

        try:
            result = tool.execute(**tool_args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error while executing tool '{tool_name}': {str(e)}"

    def _parse_response(self, raw_content: str) -> dict:
        raise NotImplementedError

    def _build_tool_result_messages(self, tool_name: str, tool_result: str) -> list[dict]:
        raise NotImplementedError

    def _parse_llm_response(self, response) -> dict:
        message = response.choices[0].message
        raw_content = getattr(message, "content", "") or ""
        return self._parse_response(raw_content)

    @staticmethod
    def _assistant_messages_from_parsed(parsed: dict) -> list[dict]:
        assistant_messages = parsed.get("assistant_messages")
        if assistant_messages:
            return list(assistant_messages)
        return [parsed["assistant_message"]]

    def _should_retry_parsed_response(self, parsed: dict) -> tuple[bool, str]:
        if parsed.get("retryable"):
            return True, str(parsed.get("retry_reason") or "retryable_parse")
        return False, ""

    @staticmethod
    def _log_retry(reason: str, *, phase: str = "run") -> None:
        print(f"[Recovery] {phase}: {reason}. Retrying with same context.")

    def _execute_parsed_tool_calls(self, step: TrajectoryStep, tool_calls: list[dict]) -> None:
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = self._execute_tool(tool_name, tool_args)
            step.tool_calls.append(
                ToolCallRecord(name=tool_name, args=tool_args, result=tool_result)
            )
            self._fire("after_tool", self.state, tool_name, tool_args, tool_result)

            for message in self._build_tool_result_messages(tool_name, tool_result):
                self.messages.append(message)

        self.state.tool_rounds += 1

    def _on_stop(self, stop_reason: str) -> Optional[str]:
        """停止条件触发时注入的最后一轮用户消息，返回 None 则不注入。
        子类可覆盖此方法自定义提示内容。"""
        prompts = {
            "max_steps":      "You have reached the maximum number of steps. Please provide your best answer now based on what you have found so far.",
            "timeout":        "Time is running out. Please provide your best answer now based on what you have found so far.",
            "max_tool_rounds": "You have used the maximum number of tool calls. Please provide your final answer now based on what you have gathered.",
            "token_budget":   "The conversation has reached the context length limit. Please provide your best answer now based on what you have found so far.",
        }
        return prompts.get(stop_reason)

    def _check_stop(self) -> str:
        """检查是否触发停止条件，返回停止原因字符串，未触发则返回空字符串。
        内置条件 + should_stop hook 均会检查，hook 优先级低于内置条件。"""
        if self.max_time_seconds and self.state.elapsed >= self.max_time_seconds:
            return "timeout"
        if self.max_tool_rounds is not None and self.state.tool_rounds >= self.max_tool_rounds:
            return "max_tool_rounds"
        hook_reason = self._fire("should_stop", self.state)
        return hook_reason or ""

    def _finalize(self, stop_reason: str) -> dict:
        """触发停止条件后，可选注入提示并再调用一次 LLM 得到最终答案。"""
        self.state.stop_reason = stop_reason
        stop_msg = self._on_stop(stop_reason)
        if not stop_msg:
            return {"final_content": f"[stopped: {stop_reason}]", "reasoning_content": "", "tool_calls": []}

        self.messages.append({"role": "user", "content": stop_msg})
        retry_budget = self._PARSE_RETRY_LIMIT

        for _ in range(self._FINALIZE_FOLLOWUP_LIMIT):
            self.state.step += 1
            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_llm_response(resp)
            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason, phase="finalize")
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(self._assistant_messages_from_parsed(parsed))
            step = TrajectoryStep(
                step=self.state.step,
                reasoning=parsed.get("reasoning_content", ""),
            )

            tool_calls = parsed["tool_calls"]
            if not tool_calls:
                step.final_content = parsed.get("final_content", "")
                self.trajectory.append(step)
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return {"final_content": f"[stopped: {stop_reason}]", "reasoning_content": "", "tool_calls": []}

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.state = RunState()
        for cm in self._context_managers:
            cm.reset()
        self.messages.append({"role": "user", "content": user_input})

        retry_budget = self._PARSE_RETRY_LIMIT
        step_idx = 0
        while step_idx < self.max_steps:
            # 每轮 LLM 调用前检查停止条件
            stop_reason = self._check_stop()
            if stop_reason:
                return self._finalize(stop_reason)

            step_idx += 1
            self.state.step = step_idx

            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_llm_response(resp)
            tool_calls = parsed["tool_calls"]
            final_content = parsed["final_content"]
            reasoning = parsed.get("reasoning_content", "")

            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason)
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(self._assistant_messages_from_parsed(parsed))
            step = TrajectoryStep(step=self.state.step, reasoning=reasoning)

            if not tool_calls:
                step.final_content = final_content
                self.trajectory.append(step)
                self.state.stop_reason = ""
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return self._finalize("max_steps")

    def run(self, user_input: str) -> str:
        parsed = self._run_loop(user_input)
        return parsed.get("final_content") or ""

    def run_verbose(self, user_input: str) -> dict:
        """返回完整解析结果：final_content、reasoning_content、tool_calls。"""
        return self._run_loop(user_input)

    def print_trajectory(self) -> None:
        """打印完整轨迹，便于调试。"""
        s = self.state
        print(f"\n[RunState] steps={s.step}  tool_rounds={s.tool_rounds}  "
              f"elapsed={s.elapsed:.1f}s  stop_reason={s.stop_reason or 'normal'}")
        for step in self.trajectory:
            print(f"\n{'='*60}")
            print(f"Step {step.step}")
            print(f"{'='*60}")
            if step.reasoning:
                print(f"[Reasoning]\n{step.reasoning}")
            if step.tool_calls:
                for tc in step.tool_calls:
                    print(f"\n[Tool Call] {tc.name}")
                    print(f"  args:   {json.dumps(tc.args, ensure_ascii=False)}")
                    print(f"  result: {tc.result}")
            if step.final_content:
                print(f"\n[Final Answer]\n{step.final_content}")

    def save_trajectory(self, path: str) -> None:
        """将完整对话轨迹（从 system 到最终回答）保存为 JSON 文件。"""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_messages, f, ensure_ascii=False, indent=2)


class NativeToolChatAgent(BaseAgent):
    """OpenAI-compatible chat.completions agent using structured tool calls."""

    def _use_native_tools(self) -> bool:
        return True

    def _default_stream(self) -> bool:
        # GLM/Kimi thinking responses can be very large. Streaming avoids provider
        # and proxy timeouts while preserving the same parsed message shape.
        return True

    @staticmethod
    def _get_field(obj: Any, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = str(item)
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _safe_load_arguments(raw_args: Any) -> Any:
        if isinstance(raw_args, (dict, list, int, float, bool)):
            return raw_args
        if raw_args is None:
            return {}
        if not isinstance(raw_args, str):
            return raw_args

        text = raw_args.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            return raw_args

    @staticmethod
    def _serialize_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments or {}, ensure_ascii=False)

    def _collect_stream_response(self, stream: Any) -> dict:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        role = "assistant"

        for chunk in stream:
            choices = self._get_field(chunk, "choices", []) or []
            for choice in choices:
                delta = self._get_field(choice, "delta", {}) or {}
                delta_role = self._get_field(delta, "role")
                if delta_role:
                    role = delta_role

                reasoning_piece = (
                    self._get_field(delta, "reasoning_content")
                    or self._get_field(delta, "reasoning")
                    or self._get_field(delta, "thinking")
                )
                if reasoning_piece:
                    reasoning_parts.append(str(reasoning_piece))

                content_piece = self._get_field(delta, "content")
                if content_piece:
                    content_parts.append(str(content_piece))

                raw_tool_calls = self._get_field(delta, "tool_calls") or []
                for fallback_index, tool_call in enumerate(raw_tool_calls):
                    index = self._get_field(tool_call, "index", fallback_index)
                    try:
                        index = int(index)
                    except Exception:
                        index = fallback_index

                    state = tool_calls_by_index.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        },
                    )

                    tool_call_id = self._get_field(tool_call, "id")
                    if tool_call_id:
                        state["id"] = tool_call_id
                    tool_call_type = self._get_field(tool_call, "type")
                    if tool_call_type:
                        state["type"] = tool_call_type

                    function = self._get_field(tool_call, "function", {}) or {}
                    tool_name = (
                        self._get_field(function, "name")
                        or self._get_field(tool_call, "name")
                    )
                    if tool_name:
                        state["function"]["name"] += str(tool_name)
                    arguments = self._get_field(function, "arguments")
                    if arguments:
                        state["function"]["arguments"] += str(arguments)

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)

        tool_calls = []
        for _, tool_call in sorted(tool_calls_by_index.items()):
            if not tool_call["function"].get("name"):
                continue
            cleaned = {
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": tool_call["function"].get("name", ""),
                    "arguments": tool_call["function"].get("arguments", ""),
                },
            }
            if tool_call.get("id"):
                cleaned["id"] = tool_call["id"]
            tool_calls.append(cleaned)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {"choices": [{"message": message}]}

    def _call_llm(self):
        raw_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages = self._apply_context_managers(raw_messages)
        messages = self._prepare_messages_for_llm(messages)
        self._persist_processed_messages(messages)
        cfg = self.llm_config

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        if cfg.tool_choice is not None:
            kwargs["tool_choice"] = copy.deepcopy(cfg.tool_choice)

        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        tools = [tool.to_schema() for tool in self.tools.values()] or None
        if tools:
            kwargs["tools"] = tools

        if self._should_stream_response():
            kwargs["stream"] = True

        kwargs = self._prepare_request_kwargs(kwargs)
        response = self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            return self._collect_stream_response(response)
        return response

    def _parse_native_tool_calls(self, raw_tool_calls: Any) -> tuple[list[dict], list[dict]]:
        if not raw_tool_calls:
            return [], []

        parsed_tool_calls: list[dict] = []
        history_tool_calls: list[dict] = []

        for tool_call in raw_tool_calls:
            function = self._get_field(tool_call, "function", {}) or {}
            tool_name = (
                self._get_field(function, "name")
                or self._get_field(tool_call, "name")
                or ""
            ).strip()
            if not tool_name:
                continue

            raw_arguments = self._get_field(function, "arguments")
            if raw_arguments is None:
                raw_arguments = self._get_field(tool_call, "arguments")

            tool_call_id = self._get_field(tool_call, "id")
            tool_call_type = self._get_field(tool_call, "type") or "function"

            parsed_tool_call = {
                "type": tool_call_type,
                "function": {
                    "name": tool_name,
                    "arguments": self._safe_load_arguments(raw_arguments),
                },
            }
            history_tool_call = {
                "type": tool_call_type,
                "function": {
                    "name": tool_name,
                    "arguments": self._serialize_arguments(raw_arguments),
                },
            }

            if tool_call_id:
                parsed_tool_call["id"] = tool_call_id
                history_tool_call["id"] = tool_call_id

            parsed_tool_calls.append(parsed_tool_call)
            history_tool_calls.append(history_tool_call)

        return parsed_tool_calls, history_tool_calls

    def _parse_response_message(self, msg: Any) -> dict:
        raw_content = self._normalize_content(self._get_field(msg, "content"))
        reasoning_content = self._normalize_content(
            self._get_field(msg, "reasoning_content")
            or self._get_field(msg, "reasoning")
            or self._get_field(msg, "thinking")
        )
        tool_calls, history_tool_calls = self._parse_native_tool_calls(
            self._get_field(msg, "tool_calls")
        )

        if tool_calls and raw_content and not reasoning_content:
            reasoning_content = raw_content

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if raw_content or tool_calls or not history_tool_calls:
            assistant_message["content"] = raw_content
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if history_tool_calls:
            assistant_message["tool_calls"] = history_tool_calls

        final_content = "" if tool_calls else raw_content

        return {
            "assistant_message": assistant_message,
            "assistant_messages": [assistant_message],
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
        }

    def _parse_llm_response(self, response) -> dict:
        choices = self._get_field(response, "choices", []) or []
        if not choices:
            return self._parse_response_message({})
        first_choice = choices[0]
        message = self._get_field(first_choice, "message", {})
        return self._parse_response_message(message)

    def _parse_response(self, raw_content: str) -> dict:
        raw_content = self._normalize_content(raw_content)
        assistant_message = {"role": "assistant", "content": raw_content}
        return {
            "assistant_message": assistant_message,
            "assistant_messages": [assistant_message],
            "tool_calls": [],
            "reasoning_content": "",
            "final_content": raw_content,
        }

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> list[dict]:
        raise NotImplementedError

    def _execute_parsed_tool_calls(self, step: TrajectoryStep, tool_calls: list[dict]) -> None:
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = self._execute_tool(tool_name, tool_args)
            step.tool_calls.append(
                ToolCallRecord(name=tool_name, args=tool_args, result=tool_result)
            )
            self._fire("after_tool", self.state, tool_name, tool_args, tool_result)

            for message in self._build_tool_result_messages(
                tool_name,
                tool_result,
                tool_call_id=tc.get("id"),
            ):
                self.messages.append(message)

        self.state.tool_rounds += 1
