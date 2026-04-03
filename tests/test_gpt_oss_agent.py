import sys
import types
import unittest
from types import SimpleNamespace


openai_stub = types.ModuleType("openai")


class OpenAI:
    def __init__(self, *args, **kwargs):
        pass


openai_stub.OpenAI = OpenAI
sys.modules.setdefault("openai", openai_stub)

from Gizmo.agents.gpt_oss_agent import GPTOssAgent
from Gizmo.tools.base_tool import BaseTool


class DummyTool(BaseTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
            },
        )

    def execute(self, **kwargs) -> str:
        return f"{self.name}:{kwargs.get('value', '')}"


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


class GPTOssAgentTemplateAlignmentTests(unittest.TestCase):
    def make_agent(self) -> GPTOssAgent:
        return GPTOssAgent(
            model="fake-model",
            api_key="fake-key",
            base_url="http://localhost",
            tools=[DummyTool("one"), DummyTool("two")],
        )

    def test_native_tool_content_is_preserved_as_reasoning(self):
        agent = self.make_agent()
        message = SimpleNamespace(
            content="Need to look this up.",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="one", arguments='{"value":"alpha"}'),
                )
            ],
        )

        parsed = agent._parse_response_message(message)

        self.assertEqual(parsed["reasoning_content"], "Need to look this up.")
        self.assertEqual(parsed["final_content"], "")
        self.assertEqual(
            parsed["assistant_messages"],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "one",
                                "arguments": {"value": "alpha"},
                            },
                            "id": "call_1",
                        }
                    ],
                    "thinking": "Need to look this up.",
                }
            ],
        )

    def test_native_tool_content_and_thinking_conflict_raises(self):
        agent = self.make_agent()
        message = SimpleNamespace(
            content="Need to look this up.",
            thinking="separate reasoning",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="one", arguments='{"value":"alpha"}'),
                )
            ],
        )

        with self.assertRaises(ValueError):
            agent._parse_response_message(message)

    def test_non_object_tool_arguments_are_preserved_in_history_shape(self):
        agent = self.make_agent()
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="one", arguments='["alpha", "beta"]'),
                )
            ],
        )

        parsed = agent._parse_response_message(message)

        self.assertEqual(
            parsed["assistant_messages"],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "one",
                                "arguments": ["alpha", "beta"],
                            },
                            "id": "call_1",
                        }
                    ],
                }
            ],
        )

    def test_raw_multi_tool_call_response_raises(self):
        agent = self.make_agent()
        raw = (
            "<|start|>assistant<|channel|>analysis<|message|>plan<|end|>"
            "<|start|>assistant to=functions.one<|channel|>commentary json<|message|>"
            '{"value":"alpha"}<|call|>'
            "<|start|>assistant to=functions.two<|channel|>commentary json<|message|>"
            '{"value":"beta"}<|call|>'
        )

        with self.assertRaises(ValueError):
            agent._parse_raw_response(raw)

    def test_prepare_messages_rejects_multiple_tool_calls_in_one_assistant_message(self):
        agent = self.make_agent()
        invalid_messages = [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "one", "arguments": {"value": "alpha"}}},
                    {"function": {"name": "two", "arguments": {"value": "beta"}}},
                ],
            },
        ]

        with self.assertRaises(ValueError):
            agent._prepare_messages_for_llm(invalid_messages)

    def test_run_loop_raises_on_multi_tool_call_response(self):
        agent = self.make_agent()
        agent.client = FakeClient(
            [
                make_response(
                    SimpleNamespace(
                        content="Need both tools.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="one",
                                    arguments='{"value":"alpha"}',
                                ),
                            ),
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(
                                    name="two",
                                    arguments='{"value":"beta"}',
                                ),
                            ),
                        ],
                    )
                ),
                make_response(SimpleNamespace(content="All done.")),
            ]
        )

        with self.assertRaises(ValueError):
            agent.run_verbose("hello")


if __name__ == "__main__":
    unittest.main()
