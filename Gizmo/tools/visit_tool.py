"""
VisitTool - online webpage visit tool aligned with the BrowseComp reference path.

Behavioral notes:
- Fetches page content through the rag.ac.cn visit proxy, not direct r.jina.ai.
- Cleans the returned markdown similarly to the reference implementation.
- Keeps the existing Gizmo tool schema: name is `visit`, arguments are `url`
  and `goal`.
"""

import json
import re
import time

import requests
import tiktoken
from openai import OpenAI

from Gizmo.prompts.tool_prompt import EXTRACTOR_PROMPT
from Gizmo.tools.base_tool import BaseTool

_VISIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "The URL(s) of the webpage(s) to visit. "
                "Can be a single URL or an array of URLs."
            ),
        },
        "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s).",
        },
    },
    "required": ["url", "goal"],
}

_VISIT_DESCRIPTION = "Visit webpage(s) and return the summary of the content."

_MAX_CONTENT_TOKENS = 95_000
_JINA_VISIT_URL = "http://api.rag.ac.cn/visit_pages_v1"
_MAX_VISIT_RETRIES = 10
_SUMMARIZER_MAX_OUTPUT_TOKENS = 8192
_CONTEXT_TOKEN_SAFETY_MARGIN = 2048
_CONTENT_REDUCTION_SAFETY_RATIO = 0.95
_MIN_RETRY_CONTENT_TOKENS = 4096
_MIN_RETRY_STEP_TOKENS = 2048
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s*(\d+)\s*tokens.*?request has\s*(\d+)\s*input tokens",
    re.IGNORECASE | re.DOTALL,
)
_OUTPUT_TOKEN_RE = re.compile(r"too large:\s*(\d+)", re.IGNORECASE)


class _RetryableVisitError(Exception):
    pass


class _ContextLengthExceededError(Exception):
    def __init__(
        self,
        *,
        max_context_tokens: int,
        input_tokens: int,
        requested_output_tokens: int,
        original_error: Exception,
    ):
        super().__init__(str(original_error))
        self.max_context_tokens = max_context_tokens
        self.input_tokens = input_tokens
        self.requested_output_tokens = requested_output_tokens
        self.original_error = original_error


def _fail_response(url: str, goal: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        "Evidence in page: \n"
        "The provided webpage content could not be accessed. "
        "Please check the URL or file format.\n\n"
        "Summary: \n"
        "The webpage content could not be processed, and therefore, "
        "no information is available.\n\n"
    )


class VisitTool(BaseTool):
    """Webpage content extraction tool backed by the rag.ac.cn visit proxy."""

    def __init__(
        self,
        jina_api_key: str,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        llm_max_retries: int = 3,
        max_content_tokens: int = _MAX_CONTENT_TOKENS,
    ):
        super().__init__(
            name="visit",
            description=_VISIT_DESCRIPTION,
            parameters=_VISIT_PARAMETERS,
        )
        self.jina_api_key = jina_api_key
        self.llm_max_retries = llm_max_retries
        self._llm = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self._llm_model = llm_model
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self.max_content_tokens = max(1, int(max_content_tokens))

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(max(1.0 * (2 ** (attempt - 1)), 2.0), 60.0)

    def _truncate_tokens(self, text: str, max_tokens: int | None = None) -> str:
        max_tokens = self.max_content_tokens if max_tokens is None else max_tokens
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @staticmethod
    def _parse_context_length_error(
        exc: Exception,
    ) -> _ContextLengthExceededError | None:
        message = str(exc)
        match = _CONTEXT_LENGTH_RE.search(message)
        if match is None:
            return None

        max_context_tokens = int(match.group(1))
        input_tokens = int(match.group(2))
        output_match = _OUTPUT_TOKEN_RE.search(message)
        requested_output_tokens = (
            int(output_match.group(1))
            if output_match is not None
            else _SUMMARIZER_MAX_OUTPUT_TOKENS
        )
        return _ContextLengthExceededError(
            max_context_tokens=max_context_tokens,
            input_tokens=input_tokens,
            requested_output_tokens=requested_output_tokens,
            original_error=exc,
        )

    def _next_retry_content_limit(
        self,
        content: str,
        current_limit: int,
        context_error: _ContextLengthExceededError,
    ) -> int:
        local_content_tokens = max(1, self._count_tokens(content))
        allowed_input_tokens = max(
            1,
            context_error.max_context_tokens
            - context_error.requested_output_tokens
            - _CONTEXT_TOKEN_SAFETY_MARGIN,
        )
        shrink_ratio = (
            allowed_input_tokens
            / max(1, context_error.input_tokens)
            * _CONTENT_REDUCTION_SAFETY_RATIO
        )
        shrink_ratio = min(0.95, shrink_ratio)
        proposed_limit = int(local_content_tokens * shrink_ratio)
        proposed_limit = min(current_limit - _MIN_RETRY_STEP_TOKENS, proposed_limit)
        return max(_MIN_RETRY_CONTENT_TOKENS, proposed_limit)

    def _fetch_jina(self, url: str) -> str:
        print(f"[Jina Reader] Reading: {url}")
        payload = {"urls": [url], "token": self.jina_api_key}
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(1, _MAX_VISIT_RETRIES + 1):
            try:
                resp = requests.post(
                    _JINA_VISIT_URL,
                    json=payload,
                    headers=headers,
                    timeout=(10, 60),
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableVisitError(f"Server Error: {resp.status_code}")
                if resp.status_code == 422:
                    raise ValueError(f"URL Unprocessable by url: {url}")

                data = resp.json()
                if data == {"error": "Invalid or expired token"}:
                    raise _RetryableVisitError("Server error: invalid or expired token")

                result_url = str((data.get("urls") or [url])[0])
                result_text = (data.get("results") or {}).get(result_url, "")
                if not isinstance(result_text, str) or not result_text.strip():
                    return "[visit] Empty content."

                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", result_text)
                text = text.replace("---", "-").replace("===", "=")
                while "   " in text:
                    text = text.replace("   ", " ")
                return text
            except ValueError:
                raise
            except (requests.RequestException, _RetryableVisitError) as exc:
                last_error = exc
                if attempt == _MAX_VISIT_RETRIES:
                    break
                print(f"Jina error, retry {attempt}/{_MAX_VISIT_RETRIES}: {exc}")
                time.sleep(self._retry_delay(attempt))
            except Exception as exc:
                last_error = exc
                break

        if last_error is None:
            return "[visit] Failed to read page."
        print(f"Jina Error: {type(last_error).__name__}: {last_error}")
        return "[visit] Failed to read page."

    def _call_llm(self, messages: list) -> str:
        content = ""
        for attempt in range(self.llm_max_retries):
            try:
                resp = self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    max_tokens=_SUMMARIZER_MAX_OUTPUT_TOKENS,
                )
                content = resp.choices[0].message.content or ""
                content = content.split("</think>")[-1]
                if content:
                    return content
            except Exception as exc:
                context_error = self._parse_context_length_error(exc)
                if context_error is not None:
                    raise context_error
                print(
                    f"Summarize Error: {type(exc).__name__}: {exc} "
                    f"content: {content!r} attempt: {attempt}"
                )
                if attempt == self.llm_max_retries - 1:
                    return content
        return content

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not isinstance(text, str):
            return None
        cleaned = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE).strip()
        for candidate in (
            cleaned,
            cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
            if "{" in cleaned and "}" in cleaned
            else "",
        ):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _summarize_single(self, url: str, content: str, goal: str) -> str:
        if (
            not content
            or content.startswith("[visit] Failed to read page.")
            or content == "[visit] Empty content."
            or content.startswith("[document_parser]")
        ):
            return _fail_response(url, goal)

        raw = ""
        content_limit = self.max_content_tokens
        while True:
            content = self._truncate_tokens(content, content_limit)
            print(
                f"call summarize {url} "
                f"(visit_content_tokens={self._count_tokens(content)}, "
                f"limit={content_limit})"
            )
            messages = [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(
                        webpage_content=content,
                        goal=goal,
                    ),
                }
            ]

            try:
                raw = self._call_llm(messages)
                break
            except _ContextLengthExceededError as exc:
                next_limit = self._next_retry_content_limit(
                    content=content,
                    current_limit=content_limit,
                    context_error=exc,
                )
                if next_limit >= content_limit:
                    return _fail_response(url, goal)
                print(
                    "[VisitTool] Context length exceeded while summarizing "
                    f"{url}; reducing visit content limit from {content_limit} "
                    f"to {next_limit} tokens and retrying."
                )
                content_limit = next_limit

        parsed = self._extract_json(raw)

        if isinstance(parsed, dict):
            try:
                result = (
                    f"The useful information in {url} for user goal {goal} as follows: \n\n"
                )
                result += f"Evidence in page: \n{parsed['evidence']}\n\n"
                result += f"Summary: \n{parsed['summary']}\n\n"
                return result
            except Exception:
                result = (
                    f"The useful information in {url} for user goal {goal} as follows: \n\n"
                )
                result += f"Summary: \n{parsed}\n\n"
                return result

        raw = raw or ""
        if len(raw) < 10:
            return _fail_response(url, goal)

        result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        result += f"Summary: \n{raw}\n\n"
        return result

    def execute(self, url, goal: str) -> str:
        try:
            urls = self.coerce_str_list(url, field_name="url", extract_urls=True)
        except ValueError as exc:
            return f"[VisitTool] {exc}"

        responses = []
        for item in urls:
            content = self._fetch_jina(item)
            responses.append(self._summarize_single(item, content, goal))
        return "\n=======\n".join(responses).strip()
