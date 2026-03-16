"""
SearchTool - 基于 Google Serper API 的在线搜索工具

功能：
- 支持批量查询（query 为字符串数组），多个 query 并行执行
- 自动检测中英文，切换对应的 location/gl/hl 参数
- 返回每条结果的标题、链接、发布日期、来源和摘要
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from Gizmo.tools.base_tool import BaseTool

_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "A list of search queries."
            ),
        }
    },
    "required": ["query"],
}

_SEARCH_DESCRIPTION = (
        "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
)


class SearchTool(BaseTool):
    """基于 Serper API 的批量 Google 搜索工具。"""

    _SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str):
        super().__init__(
            name="search",
            description=_SEARCH_DESCRIPTION,
            parameters=_SEARCH_PARAMETERS,
        )
        self.api_key = api_key

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in text)

    def _single_search(self, query: str) -> str:
        payload = {"q": query}
        if self._is_chinese(query):
            payload.update({"location": "China", "gl": "cn", "hl": "zh-cn"})
        else:
            payload.update({"location": "United States", "gl": "us", "hl": "en"})

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            resp = requests.post(
                self._SERPER_URL, headers=headers, json=payload, timeout=(3, 10)
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            return f"Google search failed for '{query}'. Error: {e}"

        if "organic" not in results:
            return f"No organic results found for query: '{query}'."

        try:
            snippets = []
            for idx, page in enumerate(results["organic"], 1):
                date_published = f"\nDate published: {page['date']}" if "date" in page else ""
                source = f"\nSource: {page['source']}" if "source" in page else ""
                snippet = f"\n{page['snippet']}" if "snippet" in page else ""
                link = page.get("link", "")
                title = page.get("title", "No Title")
                entry = f"{idx}. [{title}]({link}){date_published}{source}{snippet}"
                entry = entry.replace("Your browser can't play this video.", "")
                snippets.append(entry)

            return (
                f"A Google search for '{query}' found {len(snippets)} results:\n\n"
                f"## Web Results\n" + "\n\n".join(snippets)
            )
        except Exception as e:
            return f"Error parsing results for '{query}'. {e}"

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, query) -> str:
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            return "[SearchTool] Invalid query format: expected string or array."

        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self._single_search, q): q for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results_map[q] = future.result()
                except Exception as e:
                    results_map[q] = f"Search failed: {e}"

        return "\n=======\n".join(results_map.get(q, "Error") for q in queries)
