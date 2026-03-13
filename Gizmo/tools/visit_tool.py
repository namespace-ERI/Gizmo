"""
VisitTool - 基于 Jina Reader API 抓取网页内容，并通过辅助 LLM 提取证据和摘要

功能：
- 支持单个 URL 或 URL 数组批量处理
- goal 为必填参数，指明本次访问的目标
- 先用 Jina 拉取原始 markdown 内容（最多重试 3 次）
- 内容超过 token 限制时自动截断，LLM 输出过短时逐步缩短内容后重试
- 输出格式统一为：Evidence in page / Summary 两段
- 整批处理超过 900 秒时自动跳过剩余 URL
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
_BATCH_TIMEOUT_SECS = 900


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
    """基于 Jina + 辅助 LLM 的网页内容提取工具。"""

    def __init__(
        self,
        jina_api_key: str,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        llm_max_retries: int = 1,
    ):
        super().__init__(
            name="open_page",
            description=_VISIT_DESCRIPTION,
            parameters=_VISIT_PARAMETERS,
        )
        self.jina_api_key = jina_api_key
        self.llm_max_retries = llm_max_retries
        self._llm = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self._llm_model = llm_model
        self._encoding = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _truncate_tokens(self, text: str, max_tokens: int = _MAX_CONTENT_TOKENS) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def _fetch_jina(self, url: str) -> str:
        for attempt in range(3):
            try:
                headers = {"Authorization": f"Bearer {self.jina_api_key}"}
                resp = requests.get(
                    f"https://r.jina.ai/{url}", headers=headers, timeout=50
                )
                if resp.status_code == 200:
                    return resp.text
                raise ValueError(f"HTTP {resp.status_code}")
            except Exception:
                time.sleep(0.5)
                if attempt == 2:
                    return "[visit] Failed to read page."
        return "[visit] Failed to read page."

    def _call_llm(self, messages: list) -> str:
        for attempt in range(self.llm_max_retries):
            try:
                resp = self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.llm_max_retries - 1:
                    print(f"[VisitTool] LLM error: {e}")
                    return ""
        return ""

    def _extract_json(self, text: str) -> dict | None:
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE).strip()
        for extractor in [
            lambda t: json.loads(t),
            lambda t: json.loads(t[t.find("{") : t.rfind("}") + 1]),
        ]:
            try:
                return extractor(text)
            except Exception:
                pass
        return None

    def _summarize_single(self, url: str, content: str, goal: str) -> str:
        """对单个 URL 的内容做 LLM 抽取，失败时返回 fail_response。"""
        if not content or content.startswith("[visit] Failed to read page."):
            return _fail_response(url, goal)

        content = self._truncate_tokens(content)

        # 若 LLM 输出过短，逐步截短内容后重试（最多 3 次）
        raw = ""
        retries_left = 3
        while retries_left >= 0:
            msgs = [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(
                        webpage_content=content, goal=goal
                    ),
                }
            ]
            raw = self._call_llm(msgs)
            if len(raw) >= 10:
                break
            new_len = int(len(content) * 0.7) if retries_left > 0 else 25_000
            content = content[:new_len]
            retries_left -= 1

        # 若 JSON 解析失败，最多再重试 3 次
        parsed = self._extract_json(raw)
        if parsed is None:
            msgs = [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(
                        webpage_content=content, goal=goal
                    ),
                }
            ]
            for _ in range(3):
                raw = self._call_llm(msgs)
                parsed = self._extract_json(raw)
                if parsed is not None:
                    break

        if parsed is None:
            return _fail_response(url, goal)

        result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        result += f"Evidence in page: \n{parsed.get('evidence', '')}\n\n"
        result += f"Summary: \n{parsed.get('summary', '')}\n\n"
        return result

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, url, goal: str) -> str:
        urls = [url] if isinstance(url, str) else url

        responses = []
        start = time.time()

        for u in urls:
            if time.time() - start > _BATCH_TIMEOUT_SECS:
                responses.append(_fail_response(u, goal))
                continue
            try:
                content = self._fetch_jina(u)
                responses.append(self._summarize_single(u, content, goal))
            except Exception as e:
                responses.append(f"Error fetching {u}: {e}")

        return "\n=======\n".join(responses).strip()
