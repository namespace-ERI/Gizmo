from typing import Optional

from openai import OpenAI

from Gizmo.agents.base_agent import ContextManager, RunState

try:
    import tiktoken
except ImportError:
    tiktoken = None


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the previous conversation context so the task can continue from a "
    "compressed history. Keep the user's goal and instructions, key findings, "
    "important tool results, partial conclusions, and unresolved questions. "
    "Be concise and factual. Output only the summary."
)

SUMMARY_SECTION_MARKER = "\n\n## Previous Context Summary\n"


def _stringify_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        return str(content)
    return str(content)


def _count_tokens(messages: list[dict], encoder) -> int:
    total = 0
    for msg in messages:
        text = _stringify_content(msg.get("content"))
        if encoder is None:
            total += _estimate_tokens_without_tiktoken(text)
        else:
            total += len(encoder.encode(text))
    return total


class RollingSummaryContextManager(ContextManager):
    """Collapse the running conversation into a rolling summary when the window is full."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_input_tokens: int,
        *,
        summary_model: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        summary_max_tokens: int = 2048,
        summary_temperature: float = 0.0,
        request_timeout: float = 120.0,
        encoder_name: str = "cl100k_base",
    ):
        self._model = summary_model or model
        self._max_input_tokens = int(max_input_tokens)
        self._summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self._summary_max_tokens = int(summary_max_tokens)
        self._summary_temperature = float(summary_temperature)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(request_timeout),
        )
        self._enc = tiktoken.get_encoding(encoder_name) if tiktoken is not None else None
        self._original_user_content: Optional[str] = None
        self._rolling_summary = ""

    def reset(self) -> None:
        self._original_user_content = None
        self._rolling_summary = ""

    def process(self, messages: list[dict], state: RunState) -> list[dict]:
        if not messages or self._max_input_tokens <= 0:
            return messages

        user_idx = self._find_first_user_idx(messages)
        if user_idx is None:
            return messages

        managed = self._clone_messages(messages)
        self._remember_original_user_message(managed[user_idx])
        self._rewrite_first_user_message(managed, user_idx, self._rolling_summary)

        total_tokens = _count_tokens(managed, self._enc)
        if total_tokens <= self._max_input_tokens:
            return managed

        summary = self._summarize_messages(managed[user_idx:])
        if not summary:
            print(
                "[ContextSummary] Warning: summary generation returned empty content; "
                "keeping the original history."
            )
            return managed

        compacted = managed[: user_idx + 1]
        compacted = self._fit_summary_to_budget(compacted, user_idx, summary.strip())
        compacted_tokens = _count_tokens(compacted, self._enc)

        print(
            "[ContextSummary] Collapsed history "
            f"({total_tokens} -> {compacted_tokens} tokens, step={state.step}, "
            f"tool_rounds={state.tool_rounds})"
        )

        return compacted

    @staticmethod
    def _clone_messages(messages: list[dict]) -> list[dict]:
        return [dict(msg) for msg in messages]

    @staticmethod
    def _find_first_user_idx(messages: list[dict]) -> Optional[int]:
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                return idx
        return None

    def _remember_original_user_message(self, user_message: dict) -> None:
        if self._original_user_content is not None:
            return
        self._original_user_content = _stringify_content(user_message.get("content"))

    def _compose_first_user_content(self, summary: str) -> str:
        base = self._original_user_content or ""
        summary = (summary or "").strip()
        if not summary:
            return base
        if not base:
            return f"## Previous Context Summary\n{summary}"
        return f"{base}{SUMMARY_SECTION_MARKER}{summary}"

    def _rewrite_first_user_message(
        self,
        messages: list[dict],
        user_idx: int,
        summary: str,
    ) -> None:
        messages[user_idx]["content"] = self._compose_first_user_content(summary)

    def _summarize_messages(self, messages: list[dict]) -> str:
        rendered = self._render_messages(messages)
        if not rendered:
            return ""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._summary_prompt},
                {"role": "user", "content": rendered},
            ],
            temperature=self._summary_temperature,
            max_tokens=self._summary_max_tokens,
        )
        return _stringify_content(response.choices[0].message.content).strip()

    @staticmethod
    def _render_messages(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role") or "unknown"
            content = _stringify_content(msg.get("content")).strip()
            if not content:
                continue
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts).strip()

    def _fit_summary_to_budget(
        self,
        messages: list[dict],
        user_idx: int,
        summary: str,
    ) -> list[dict]:
        fitted = self._clone_messages(messages)
        self._rewrite_first_user_message(fitted, user_idx, summary)
        if _count_tokens(fitted, self._enc) <= self._max_input_tokens:
            self._rolling_summary = summary
            return fitted

        if self._enc is None:
            low, high = 0, len(summary)
            candidate_from_mid = lambda mid: summary[:mid].strip()
        else:
            summary_tokens = self._enc.encode(summary)
            low, high = 0, len(summary_tokens)
            candidate_from_mid = lambda mid: self._enc.decode(summary_tokens[:mid]).strip()
        best = ""

        while low <= high:
            mid = (low + high) // 2
            candidate_summary = candidate_from_mid(mid)
            candidate_messages = self._clone_messages(messages)
            self._rewrite_first_user_message(candidate_messages, user_idx, candidate_summary)
            if _count_tokens(candidate_messages, self._enc) <= self._max_input_tokens:
                best = candidate_summary
                low = mid + 1
            else:
                high = mid - 1

        self._rolling_summary = best
        self._rewrite_first_user_message(fitted, user_idx, best)

        if _count_tokens(fitted, self._enc) > self._max_input_tokens:
            print(
                "[ContextSummary] Warning: compacted prompt still exceeds the token budget "
                "after trimming the summary."
            )

        return fitted


def _estimate_tokens_without_tiktoken(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
