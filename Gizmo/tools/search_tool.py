import time
from urllib.parse import urlparse

import requests
import tiktoken
from requests.exceptions import Timeout

from Gizmo.tools.base_tool import BaseTool

_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query string",
        }
    },
    "required": ["query"],
}

_SEARCH_DESCRIPTION = (
    "Performs a web search. The tool retrieves the top 10 results for the query, "
    "returning their docid (temporary index, keep unique in the main session), "
    "source domain, and document snippet (may be truncated based on token limits)."
)


class SearchTool(BaseTool):
    """Google Custom Search tool."""

    def __init__(self, api_key: str, cx: str, snippet_max_tokens: int = 128):
        super().__init__(
            name="search",
            description=_SEARCH_DESCRIPTION,
            parameters=_SEARCH_PARAMETERS,
        )
        self.api_key = api_key
        self.cx = cx
        self.snippet_max_tokens = snippet_max_tokens
        self._api_url = "https://www.googleapis.com/customsearch/v1"
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._docid_counter = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        return self._encoding.decode(tokens[: self.snippet_max_tokens])

    def _next_docid(self) -> int:
        self._docid_counter += 1
        return self._docid_counter

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc or url

    def _raw_search(self, query: str, timeout: int = 20) -> dict:
        params = {"q": query, "key": self.api_key, "cx": self.cx}
        for attempt in range(1, 4):
            try:
                resp = requests.get(self._api_url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Timeout:
                if attempt == 3:
                    print(f"[SearchTool] Timeout for query: {query!r} after 3 retries")
                    return {}
                print(f"[SearchTool] Timeout, retrying ({attempt}/3)...")
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    print(f"[SearchTool] Request error: {e} after 3 retries")
                    return {}
                print(f"[SearchTool] Request error: {e}, retrying ({attempt}/3)...")
            time.sleep(1)
        return {}

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, query: str) -> str:
        data = self._raw_search(query)

        if not data or "items" not in data:
            return f"[SearchTool] No results for query: {query!r}"

        lines = [f'Search results for: "{query}"\n']
        seen_urls: set[str] = set()

        for item in data.get("items", []):
            url = item.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            docid = self._next_docid()
            domain = self._domain(url)
            snippet = self._truncate(item.get("snippet", ""))
            title = item.get("title", "")

            lines.append(
                f"[{docid}] {title}\n"
                f"    Source: {domain}\n"
                f"    Snippet: {snippet}\n"
            )

        return "\n".join(lines)
