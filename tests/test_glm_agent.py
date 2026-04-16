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
from Gizmo.agents.glm_agent import GLMAgent
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


class GLMAgentCompatibilityTests(unittest.TestCase):
    def make_agent(self, *, llm_config: LLMConfig | None = None, max_steps: int = 3) -> GLMAgent:
        return GLMAgent(
            model="glm-4.5",
            api_key="fake-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            tools=[DummyTool("search")],
            llm_config=llm_config or LLMConfig(),
            max_steps=max_steps,
        )

    def test_call_llm_uses_chat_tools_shape_and_glm_thinking_body(self):
        agent = self.make_agent(
            llm_config=LLMConfig(
                max_tokens=256,
                temperature=0.2,
                top_p=0.8,
                tool_choice="auto",
                enable_thinking=True,
            )
        )
        agent.client = FakeClient([make_response(SimpleNamespace(content="done"))])
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        payload = agent.client.chat.completions.calls[0]
        self.assertEqual(payload["model"], "glm-4.5")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["tool_choice"], "auto")
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
        self.assertEqual(
            payload["extra_body"],
            {"thinking": {"type": "enabled", "clear_thinking": False}},
        )

    def test_parse_response_message_preserves_official_tool_history_shape(self):
        agent = self.make_agent()
        message = SimpleNamespace(
            content="",
            reasoning_content="Need to search first.",
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

        parsed = agent._parse_response_message(message)

        self.assertEqual(parsed["reasoning_content"], "Need to search first.")
        self.assertEqual(parsed["final_content"], "")
        self.assertEqual(
            parsed["assistant_message"],
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need to search first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"alpha"}',
                        },
                    }
                ],
            },
        )
        self.assertEqual(
            parsed["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "alpha"},
                    },
                }
            ],
        )

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
