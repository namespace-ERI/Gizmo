import copy
import os
from typing import Optional

from Gizmo.agents.base_agent import NativeToolChatAgent
from Gizmo.prompts.system_prompt import KIMI_SYSTEM_PROMPT


class KimiAgent(NativeToolChatAgent):
    """Kimi 官方 OpenAI 兼容 chat/tool-calling 适配器。"""

    def __init__(
        self,
        *args,
        system_prompt: str = KIMI_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            system_prompt=system_prompt,
            base_url=(
                base_url
                or os.environ.get("KIMI_BASE_URL")
                or os.environ.get("TRANSIT_BASE_URL")
                or "http://localhost:8001/v1"
            ),
            **kwargs,
        )

    def _supports_thinking_parameter(self) -> bool:
        model_name = str(self.model or "").strip().lower()
        return model_name == "kimi-k2.5"

    def _build_extra_body(self) -> Optional[dict]:
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        if cfg.enable_thinking and self._supports_thinking_parameter():
            thinking = base.setdefault("thinking", {})
            thinking.setdefault("type", "enabled")

        return base or None

    def _prepare_request_kwargs(self, kwargs: dict) -> dict:
        if str(self.model or "").strip().lower() == "kimi-k2.5":
            # Kimi K2.5 fixes these sampling parameters; omitting them avoids
            # provider-side validation errors while preserving the official defaults.
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        return kwargs

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> list[dict]:
        message = {
            "role": "tool",
            "name": tool_name,
            "content": tool_result,
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return [message]
