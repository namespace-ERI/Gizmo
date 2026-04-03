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


class _RetryableVisitError(Exception):
    pass


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

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(max(1.0 * (2 ** (attempt - 1)), 2.0), 60.0)

    def _truncate_tokens(self, text: str, max_tokens: int = _MAX_CONTENT_TOKENS) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

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
                    max_tokens=8192,
                )
                content = resp.choices[0].message.content or ""
                content = content.split("</think>")[-1]
                if content:
                    return content
            except Exception as exc:
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

        content = self._truncate_tokens(content)
        print(f"call summarize {url}")
        messages = [
            {
                "role": "user",
                "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
            }
        ]

        raw = self._call_llm(messages)
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
