import copy
import json
from typing import Any, Optional

from Gizmo.agents.base_agent import (
    BaseAgent,
    RunState,
    ToolCallRecord,
    TrajectoryStep,
)
from Gizmo.prompts.system_prompt import KIMI_SYSTEM_PROMPT


class KimiAgent(BaseAgent):
    """Kimi 官方 OpenAI 兼容 chat/tool-calling 适配器。"""

    def __init__(
        self,
        *args,
        system_prompt: str = KIMI_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            system_prompt=system_prompt,
            base_url=base_url or "https://api.moonshot.cn/v1",
            **kwargs,
        )

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

    def _build_extra_body(self) -> Optional[dict]:
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        if cfg.enable_thinking:
            thinking = base.setdefault("thinking", {})
            thinking.setdefault("type", "enabled")

        return base or None

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

        return self.client.chat.completions.create(**kwargs)

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

    def _parse_response(self, raw_content: str) -> dict:
        raw_content = self._normalize_content(raw_content)
        return {
            "assistant_message": {"role": "assistant", "content": raw_content},
            "assistant_messages": [{"role": "assistant", "content": raw_content}],
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
        message = {
            "role": "tool",
            "name": tool_name,
            "content": tool_result,
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return [message]

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

    def _finalize(self, stop_reason: str) -> dict:
        self.state.stop_reason = stop_reason
        stop_msg = self._on_stop(stop_reason)
        if not stop_msg:
            return {
                "final_content": f"[stopped: {stop_reason}]",
                "reasoning_content": "",
                "tool_calls": [],
            }

        self.messages.append({"role": "user", "content": stop_msg})
        retry_budget = self._PARSE_RETRY_LIMIT

        for _ in range(self._FINALIZE_FOLLOWUP_LIMIT):
            self.state.step += 1
            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_response_message(resp.choices[0].message)
            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason, phase="finalize")
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(parsed.get("assistant_messages") or [parsed["assistant_message"]])
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

        return {
            "final_content": f"[stopped: {stop_reason}]",
            "reasoning_content": "",
            "tool_calls": [],
        }

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.state = RunState()
        for cm in self._context_managers:
            cm.reset()
        self.messages.append({"role": "user", "content": user_input})

        retry_budget = self._PARSE_RETRY_LIMIT

        for step_idx in range(self.max_steps):
            stop_reason = self._check_stop()
            if stop_reason:
                return self._finalize(stop_reason)

            self.state.step = step_idx + 1

            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_response_message(resp.choices[0].message)
            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason)
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            assistant_messages = parsed.get("assistant_messages") or [parsed["assistant_message"]]
            tool_calls = parsed["tool_calls"]
            reasoning = parsed.get("reasoning_content", "")
            final_content = parsed.get("final_content", "")

            step = TrajectoryStep(step=self.state.step, reasoning=reasoning)
            self.messages.extend(assistant_messages)

            if not tool_calls:
                step.final_content = final_content
                self.trajectory.append(step)
                self.state.stop_reason = ""
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return self._finalize("max_steps")
