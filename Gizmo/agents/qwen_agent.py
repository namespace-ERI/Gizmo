import json
import re

from Gizmo.agents.base_agent import BaseAgent
from Gizmo.prompts.system_prompt import QWEN_SYSTEM_PROMPT


class QwenAgent(BaseAgent):
    def __init__(self, *args, system_prompt: str = QWEN_SYSTEM_PROMPT, **kwargs):
        super().__init__(*args, system_prompt=system_prompt, **kwargs)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_blocks = []
        for tool in self.tools.values():
            tool_blocks.append(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        tools_text = "\n".join(tool_blocks)

        if "{tool_des}" in self.system_prompt:
            return self.system_prompt.format(tool_des=f"<tools>\n{tools_text}\n</tools>" if tools_text else "")

        if not tools_text:
            return self.system_prompt

        tool_instruction = f"""# Tools

You have access to the following functions:

<tools>
{tools_text}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>""".strip()

        return f"{self.system_prompt}\n\n{tool_instruction}"

    def _parse_reasoning_content(self, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        # 兼容两种情况：完整的 <think>...</think>，以及只有 </think> 结尾（无开头标签，因为qwen3.5很多推理内容没有开头标签）
        think_pattern = re.compile(
            r"<think>\s*(.*?)\s*</think>",
            re.IGNORECASE | re.DOTALL,
        )
        matches = think_pattern.findall(content)
        if matches:
            return "\n\n".join(m.strip() for m in matches if m.strip())

        close_tag = re.search(r"</think>", content, re.IGNORECASE)
        if close_tag:
            return content[: close_tag.start()].strip()

        return ""

    def _remove_reasoning_content(self, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        cleaned = re.sub(
            r"<think>\s*.*?\s*</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cleaned != content:
            return cleaned.strip()

        # 没有开头 <think>，但存在 </think>：把 </think> 之前的内容都视为推理
        close_tag = re.search(r"</think>", content, re.IGNORECASE)
        if close_tag:
            return content[close_tag.end() :].strip()

        return cleaned.strip()

    def _parse_tool_calls(self, content: str) -> list[dict]:
        content = (content or "").strip()
        if not content:
            return []

        tool_pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.IGNORECASE | re.DOTALL,
        )
        function_pattern = re.compile(
            r"<function=([^\n>]+)>\s*(.*?)\s*</function>",
            re.IGNORECASE | re.DOTALL,
        )
        parameter_pattern = re.compile(
            r"<parameter=([^\n>]+)>\s*(.*?)\s*</parameter>",
            re.IGNORECASE | re.DOTALL,
        )

        results = []

        tool_blocks = tool_pattern.findall(content)
        for tool_block in tool_blocks:
            function_match = function_pattern.search(tool_block)
            if not function_match:
                continue

            tool_name = function_match.group(1).strip()
            function_body = function_match.group(2)

            arguments = {}
            for param_name, param_value in parameter_pattern.findall(function_body):
                arguments[param_name.strip()] = param_value.strip()

            results.append(
                {
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                }
            )

        return results

    def _remove_tool_calls(self, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        cleaned = re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    def _extract_final_content(self, raw_content: str) -> str:
        content = self._remove_reasoning_content(raw_content)
        content = self._remove_tool_calls(content)
        return content.strip()

    def _parse_response(self, raw_content: str) -> dict:
        tool_calls = self._parse_tool_calls(raw_content)
        reasoning_content = self._parse_reasoning_content(raw_content)
        final_content = self._extract_final_content(raw_content)

        # 只保留标准 message 字段，避免非法字段传回 API
        # 同时保留 raw_content，让模型下一轮还能看到自己刚才的原始 <tool_call>
        assistant_message = {
            "role": "assistant",
            "content": raw_content,
        }

        return {
            "assistant_message": assistant_message,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
        }

    def _build_tool_result_messages(self, tool_name: str, tool_result: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": (
                    f"<tool_response>\n{tool_result}\n</tool_response>\n"
                    f"Please continue."
                ),
            }
        ]