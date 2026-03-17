"""
LocalVisitTool - 基于本地语料库缓存 + vllm LLM API 的离线文档访问工具

功能：
- 仅支持通过 URL（LocalSearchTool 搜索后结果中的 URL）访问文档
- 文档内容直接从 LocalSearchTool 的内存缓存读取，不走任何在线接口
- URL 无效（未搜索过或不存在）时直接返回错误，不做保底降级
- 支持单个 URL 或 URL 数组，共用同一个 goal 批量提取
- goal 为必填参数，始终调用本地 vllm LLM API 根据 goal 抽取 evidence/summary

初始化参数：
    llm_api_key:  vllm LLM API 密钥（本地部署通常为 "EMPTY"）
    llm_base_url: vllm LLM API 地址（如 http://localhost:8001/v1）
    llm_model:    LLM 模型名称
    search_tool:  已初始化的 LocalSearchTool 实例（提供 docid 缓存）
"""

import json

from Gizmo.prompts.tool_prompt import EXTRACTOR_PROMPT
from Gizmo.tools.base_tool import BaseTool
from openai import OpenAI

_LOCAL_VISIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
    },
    "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
    }
    },
    "required": ["url", "goal"]
}

_LOCAL_VISIT_DESCRIPTION = (
    "Visit webpage(s) and return the summary of the content."
)


class LocalVisitTool(BaseTool):
    """Visit tool backed purely by the local corpus cache (no Jina fallback).

    Content is retrieved by URL from the linked LocalSearchTool's in-memory corpus
    store. Only URLs that appeared in a prior search result are accessible; if the
    URL is not found, an error is returned immediately.

    Args:
        llm_api_key:       API key for the auxiliary LLM (usually "EMPTY" for local vllm).
        llm_base_url:      Base URL of the vllm LLM API.
        llm_model:         Model name to use for evidence extraction.
        search_tool:       The LocalSearchTool instance whose corpus cache is used.
        llm_max_retries:   Number of retries on LLM call failure.
        max_content_chars: Maximum characters of corpus text sent to the LLM.
    """

    MAX_CONTENT_CHARS = 100000

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        search_tool,
        llm_max_retries: int = 2,
        max_content_chars: int = MAX_CONTENT_CHARS,
    ):
        super().__init__(
            name="visit",
            description=_LOCAL_VISIT_DESCRIPTION,
            parameters=_LOCAL_VISIT_PARAMETERS,
        )
        self.search_tool = search_tool
        self.llm_max_retries = llm_max_retries
        self.max_content_chars = max_content_chars
        self._llm = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self._llm_model = llm_model

    # ------------------------------------------------------------------
    # evidence extraction
    # ------------------------------------------------------------------

    def _extract_evidence(self, content: str, goal: str) -> str:
        messages = [
            {
                "role": "user",
                "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
            }
        ]
        for attempt in range(self.llm_max_retries):
            try:
                resp = self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    temperature=0.7,
                )
                raw = resp.choices[0].message.content or ""
                try:
                    json.loads(raw)
                    return raw
                except Exception:
                    left, right = raw.find("{"), raw.rfind("}")
                    if left != -1 and right != -1 and left <= right:
                        return raw[left : right + 1]
                    return raw
            except Exception as e:
                if attempt == self.llm_max_retries - 1:
                    return f"[LocalVisitTool] LLM extraction error: {e}"
        return ""

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, url, goal: str) -> str:
        try:
            urls = self.coerce_str_list(url, field_name="url", extract_urls=True)
        except ValueError as e:
            return f"[LocalVisitTool] {e}"

        results = []
        for u in urls:
            text = self.search_tool.get_text_by_url(u)
            if text is None:
                results.append(
                    f"[LocalVisitTool] URL not found in corpus cache: {u!r}. "
                    "Make sure to run a search first and use a URL from the search results."
                )
                continue

            content = text[: self.max_content_chars]
            evidence = self._extract_evidence(content, goal)
            results.append(f"URL: {u}\nEvidence:\n{evidence}")

        return "\n=======\n".join(results)
