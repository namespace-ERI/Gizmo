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

from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.base_tool import BaseTool


class DummyTool(BaseTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs) -> str:
        return f"{self.name}:{kwargs.get('query')}"


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


def make_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class QwenAgentRecoveryTests(unittest.TestCase):
    def make_agent(self, *, max_steps: int = 3) -> QwenAgent:
        return QwenAgent(
            model="fake-model",
            api_key="fake-key",
            base_url="http://localhost",
            tools=[DummyTool("search")],
            max_steps=max_steps,
        )

    def test_parse_response_marks_malformed_tool_markup_retryable(self):
        agent = self.make_agent()
        raw = (
            "I need to trace through this research chain step by step.\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        parsed = agent._parse_response(raw)

        self.assertEqual(parsed["tool_calls"], [])
        self.assertTrue(parsed["retryable"])
        self.assertEqual(parsed["retry_reason"], "malformed_tool_call_xml")
        self.assertEqual(parsed["final_content"], "")

    def test_run_loop_retries_malformed_tool_markup_once(self):
        agent = self.make_agent(max_steps=3)
        agent.client = FakeClient(
            [
                make_response(
                    "I should search for the source first.\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response(
                    "<think>Search first.</think>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    "[\"alpha\"]\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response("Final answer."),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 3)
        self.assertEqual(agent.state.tool_rounds, 1)
        self.assertEqual(len(agent.messages), 4)

    def test_finalize_executes_tool_call_after_stop_prompt(self):
        agent = self.make_agent(max_steps=1)
        agent.client = FakeClient(
            [
                make_response(
                    "<think>Need an initial search.</think>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    "[\"alpha\"]\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response(
                    "<think>Need one more search before answering.</think>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    "[\"beta\"]\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response("Final answer."),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 3)
        self.assertEqual(agent.state.tool_rounds, 2)
        self.assertTrue(
            any(
                msg.get("role") == "user"
                and "maximum number of steps" in msg.get("content", "")
                for msg in agent.messages
            )
        )
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertEqual(agent.messages[-1]["content"], "Final answer.")


if __name__ == "__main__":
    unittest.main()
