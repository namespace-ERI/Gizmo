"""
SearchTool - online search tool aligned with the BrowseComp reference path.

Behavioral notes:
- Uses the rag.ac.cn SERP proxy instead of the direct Serper endpoint.
- Keeps the existing Gizmo tool schema: name is `search`, argument is `query`.
- Accepts one or more queries and returns the same textual result format as the
  reference implementation.
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
            "description": "A list of search queries.",
        }
    },
    "required": ["query"],
}

_SEARCH_DESCRIPTION = (
    "Performs batched web searches: supply an array 'query'; the tool retrieves "
    "the top results for each query in one call."
)

_SERP_URL = "http://api2.rag.ac.cn/serp_search_v1"
_MAX_SEARCH_RETRIES = 8


class _RetryableSearchError(Exception):
    pass


class SearchTool(BaseTool):
    """Batch Google search tool backed by the rag.ac.cn SERP proxy."""

    def __init__(self, api_key: str):
        super().__init__(
            name="search",
            description=_SEARCH_DESCRIPTION,
            parameters=_SEARCH_PARAMETERS,
        )
        self.api_key = api_key

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in text)

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(max(0.5 * (2 ** (attempt - 1)), 1.0), 5.0)

    def _build_payload(self, query: str) -> dict:
        payload = {
            "query": query,
            "page": 1,
            "use_cache": True,
            "token": self.api_key,
            "search_type": "search",
        }
        if self._is_chinese(query):
            payload.update({"location": "China", "gl": "cn", "hl": "zh-cn"})
        else:
            payload.update({"location": "United States", "gl": "us", "hl": "en"})
        return payload

    def _request_results(self, query: str) -> dict:
        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(query)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_SEARCH_RETRIES + 1):
            print(f"[Serper] Searching: {query} 1")
            try:
                resp = requests.post(
                    _SERP_URL,
                    json=payload,
                    headers=headers,
                    timeout=(5, 30),
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableSearchError(f"Server Error: {resp.status_code}")
                if resp.status_code == 422:
                    raise ValueError(f"URL Unprocessable by Serper: {query}")

                data = resp.json()
                if data == {"error": "Invalid or expired token"}:
                    raise _RetryableSearchError("Server error: invalid or expired token")
                if "organic" not in data:
                    raise _RetryableSearchError("Error, No results found for query")
                return data
            except ValueError:
                raise
            except (requests.RequestException, _RetryableSearchError) as exc:
                last_error = exc
                if attempt == _MAX_SEARCH_RETRIES:
                    break
                print(f"Serper error, retry {attempt}/{_MAX_SEARCH_RETRIES}: {exc}")
                time.sleep(self._retry_delay(attempt))
            except Exception as exc:
                last_error = exc
                break

        if last_error is None:
            last_error = RuntimeError("Unknown search failure")
        raise last_error

    def _format_results(self, query: str, results: dict) -> str:
        web_snippets: list[str] = []
        for idx, page in enumerate(results["organic"], 1):
            date_published = (
                "\nDate published: " + str(page["date"]) if "date" in page else ""
            )
            source = "\nSource: " + str(page["source"]) if "source" in page else ""
            snippet = "\n" + str(page["snippet"]) if "snippet" in page else ""
            title = str(page.get("title", "No Title"))
            link = str(page.get("link", ""))
            redacted = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
            redacted = redacted.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted)

        return (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n"
            "## Web Results\n"
            + "\n\n".join(web_snippets)
        )

    def _single_search(self, query: str) -> str:
        try:
            results = self._request_results(query)
            if "organic" not in results:
                return (
                    f"No results found for query: '{query}'. "
                    "Try with a more general query."
                )
            return self._format_results(query, results)
        except ValueError:
            print("Http Error. Retry next time.")
            return "Http Error. Retry next time."
        except Exception as exc:
            print(f"Serper Error: {type(exc).__name__}: {exc}")
            return f"Search Error: {type(exc).__name__}: {exc}"

    def execute(self, query) -> str:
        try:
            queries = self.coerce_str_list(query, field_name="query")
        except ValueError as exc:
            return f"[SearchTool] {exc}"

        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self._single_search, q): q for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results_map[q] = future.result()
                except Exception as exc:
                    results_map[q] = f"Search Error: {type(exc).__name__}: {exc}"

        return "\n=======\n".join(results_map.get(q, "Error") for q in queries)
