import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


openai_stub = types.ModuleType("openai")


class OpenAI:
    def __init__(self, *args, **kwargs):
        pass


openai_stub.OpenAI = OpenAI
sys.modules.setdefault("openai", openai_stub)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.deepseek_agent import DeepSeekAgent
from Gizmo.tools.base_tool import BaseTool


class DummyTool(BaseTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs) -> str:
        return f"{self.name}:{kwargs.get('query', '')}"


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake LLM responses left.")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DeepSeekAgentCompatibilityTests(unittest.TestCase):
    def make_agent(
        self,
        *,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        llm_config: LLMConfig | None = None,
        max_steps: int = 3,
    ) -> DeepSeekAgent:
        return DeepSeekAgent(
            model=model,
            api_key="fake-key",
            base_url=base_url,
            tools=[DummyTool("search")],
            llm_config=llm_config or LLMConfig(stream=False),
            max_steps=max_steps,
        )

    def test_call_llm_uses_deepseek_thinking_mode(self):
        agent = self.make_agent(
            llm_config=LLMConfig(
                max_tokens=256,
                temperature=0.2,
                top_p=0.8,
                tool_choice="auto",
                enable_thinking=True,
                stream=False,
            )
        )
        agent.client = FakeClient([make_response(SimpleNamespace(content="done"))])
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        payload = agent.client.chat.completions.calls[0]
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_call_llm_passes_explicit_deepseek_reasoning_effort(self):
        agent = self.make_agent(
            llm_config=LLMConfig(
                enable_thinking=True,
                reasoning_effort="max",
                stream=False,
            )
        )
        agent.client = FakeClient([make_response(SimpleNamespace(content="done"))])
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        payload = agent.client.chat.completions.calls[0]
        self.assertEqual(
            payload["extra_body"], {"thinking": {"type": "enabled"}}
        )
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "search tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

    def test_call_llm_disables_deepseek_thinking_when_configured_off(self):
        agent = self.make_agent(llm_config=LLMConfig(enable_thinking=False, stream=False))
        agent.client = FakeClient([make_response(SimpleNamespace(content="done"))])
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        payload = agent.client.chat.completions.calls[0]
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", payload)

    def test_run_loop_executes_tool_and_appends_tool_message_with_tool_call_id(self):
        agent = self.make_agent(max_steps=3)
        agent.client = FakeClient(
            [
                make_response(
                    SimpleNamespace(
                        content="",
                        reasoning_content="Need evidence.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="search",
                                    arguments='{"query":"alpha"}',
                                ),
                            )
                        ],
                    )
                ),
                make_response(SimpleNamespace(content="Final answer.")),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 2)
        self.assertEqual(agent.state.tool_rounds, 1)
        self.assertEqual(agent.trajectory[0].reasoning, "Need evidence.")
        self.assertTrue(
            any(
                msg.get("role") == "tool"
                and msg.get("tool_call_id") == "call_1"
                and msg.get("content") == "search:alpha"
                for msg in agent.messages
            )
        )


if __name__ == "__main__":
    unittest.main()
