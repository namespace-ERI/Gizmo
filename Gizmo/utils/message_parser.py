import json
import re
from typing import Any


_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def normalize_content(content: Any) -> str:
    """Convert assistant content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _build_tool_call(function_name: str, arguments: dict, index: int) -> dict:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def extract_reasoning_content(content: Any) -> tuple[str, str]:
    """Extract <think>...</think> and return (reasoning_content, remaining_content)."""
    raw_text = normalize_content(content)
    if not raw_text:
        return "", ""

    reasoning_parts = [
        match.group(1).strip()
        for match in _THINK_BLOCK_RE.finditer(raw_text)
        if match.group(1).strip()
    ]
    reasoning_content = "\n\n".join(reasoning_parts)
    remaining_content = _THINK_BLOCK_RE.sub("", raw_text)
    return reasoning_content, _clean_text(remaining_content)


def _parse_json_style_tool_call(tool_call_body: str) -> tuple[str, dict] | None:
    """Fallback parser for <tool_call>{...}</tool_call> style payload."""
    text = tool_call_body.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload.get("name"):
        return None

    function_name = str(payload["name"]).strip()
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return function_name, arguments


def extract_tool_calls(content: Any, start_index: int = 1) -> tuple[list[dict], str]:
    """Extract <tool_call> blocks and return (tool_calls, remaining_content)."""
    raw_text = normalize_content(content)
    if not raw_text:
        return [], ""

    tool_calls: list[dict] = []
    next_index = start_index

    for tool_call_match in _TOOL_CALL_BLOCK_RE.finditer(raw_text):
        tool_call_body = tool_call_match.group(1)
        function_matches = list(_FUNCTION_BLOCK_RE.finditer(tool_call_body))

        if function_matches:
            for function_match in function_matches:
                function_name = function_match.group(1).strip()
                function_body = function_match.group(2)
                if not function_name:
                    continue

                arguments: dict[str, Any] = {}
                for parameter_match in _PARAMETER_BLOCK_RE.finditer(function_body):
                    param_name = parameter_match.group(1).strip()
                    if not param_name:
                        continue
                    arguments[param_name] = _parse_parameter_value(parameter_match.group(2))

                tool_calls.append(_build_tool_call(function_name, arguments, next_index))
                next_index += 1
            continue

        parsed = _parse_json_style_tool_call(tool_call_body)
        if parsed:
            function_name, arguments = parsed
            tool_calls.append(_build_tool_call(function_name, arguments, next_index))
            next_index += 1

    remaining_content = _TOOL_CALL_BLOCK_RE.sub("", raw_text)
    return tool_calls, _clean_text(remaining_content)
