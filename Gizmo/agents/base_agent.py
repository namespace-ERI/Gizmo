import json
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

"""
    对所有Agent的基类，提供了LLM调用、工具调用、解析响应、执行工具等基本功能。不同厂商的template不同所以为了适配不同模型，需要重写一些方法。
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
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[list] = None,
        max_steps: int = 200,
        timeout: float = 120.0,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.messages: list[dict] = []
        self.trajectory: list[TrajectoryStep] = []

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _use_native_tools(self) -> bool:
        return False

    def _call_llm(self):
        messages = [{"role": "system", "content": self.system_prompt}] + self.messages

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

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

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.messages.append({"role": "user", "content": user_input})

        for step_idx in range(self.max_steps):
            resp = self._call_llm()
            msg = resp.choices[0].message
            raw_content = msg.content or ""

            parsed = self._parse_response(raw_content)
            assistant_message = parsed["assistant_message"]
            tool_calls = parsed["tool_calls"]
            final_content = parsed["final_content"]
            reasoning = parsed.get("reasoning_content", "")

            self.messages.append(assistant_message)

            step = TrajectoryStep(step=step_idx + 1, reasoning=reasoning)

            if not tool_calls:
                step.final_content = final_content
                self.trajectory.append(step)
                return parsed

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"].get("arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                tool_result = self._execute_tool(tool_name, tool_args)
                step.tool_calls.append(ToolCallRecord(name=tool_name, args=tool_args, result=tool_result))

                for message in self._build_tool_result_messages(tool_name, tool_result):
                    self.messages.append(message)

            self.trajectory.append(step)

        return {"final_content": "[reached max steps]", "reasoning_content": "", "tool_calls": []}

    def run(self, user_input: str) -> str:
        parsed = self._run_loop(user_input)
        return parsed.get("final_content") or ""

    def run_verbose(self, user_input: str) -> dict:
        """返回完整解析结果：final_content、reasoning_content、tool_calls。"""
        return self._run_loop(user_input)

    def print_trajectory(self) -> None:
        """打印完整轨迹，便于调试。"""
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