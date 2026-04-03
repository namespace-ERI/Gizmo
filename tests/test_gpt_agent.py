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
from Gizmo.agents.gpt_agent import GPTAgent
from Gizmo.tools.base_tool import BaseTool


class DummyTool(BaseTool):
    strict = True

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


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake Responses API payload left.")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


def make_response(*, output, output_text="", response_id="resp_1", status="completed"):
    return SimpleNamespace(
        id=response_id,
        status=status,
        output=output,
        output_text=output_text,
    )


class GPTAgentResponsesTests(unittest.TestCase):
    def make_agent(self, *, max_steps: int = 3, llm_config: LLMConfig | None = None) -> GPTAgent:
        return GPTAgent(
            model="gpt-5-mini",
            api_key="fake-key",
            tools=[DummyTool("search")],
            max_steps=max_steps,
            llm_config=llm_config or LLMConfig(),
        )

    def test_call_llm_uses_responses_api_shape(self):
        agent = self.make_agent(
            llm_config=LLMConfig(
                max_output_tokens=256,
                temperature=0.2,
                top_p=0.9,
                parallel_tool_calls=True,
                tool_choice="auto",
                reasoning_effort="medium",
                reasoning_summary="auto",
                text_verbosity="low",
                include=["reasoning.summary"],
                metadata={"run": "test"},
            )
        )
        agent.client = FakeClient(
            [
                make_response(
                    output=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ],
                    output_text="done",
                )
            ]
        )
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        self.assertEqual(len(agent.client.responses.calls), 1)
        payload = agent.client.responses.calls[0]
        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["instructions"], agent.system_prompt)
        self.assertEqual(
            payload["input"],
            [{"type": "message", "role": "user", "content": "hello"}],
        )
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["reasoning"], {"effort": "medium", "summary": "auto"})
        self.assertEqual(payload["text"], {"verbosity": "low"})
        self.assertEqual(payload["include"], ["reasoning.summary"])
        self.assertEqual(payload["metadata"], {"run": "test"})
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "name": "search",
                    "description": "search tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "strict": True,
                }
            ],
        )

    def test_parse_response_keeps_message_text_as_reasoning_when_tool_call_present(self):
        agent = self.make_agent()

        parsed = agent._parse_response(
            make_response(
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Need to search first."}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": '{"query":"alpha"}',
                    },
                ]
            )
        )

        self.assertEqual(parsed["final_content"], "")
        self.assertEqual(parsed["reasoning_content"], "Need to search first.")
        self.assertEqual(
            parsed["tool_calls"],
            [
                {
                    "id": "fc_1",
                    "call_id": "call_1",
                    "status": None,
                    "function": {
                        "name": "search",
                        "arguments": {"query": "alpha"},
                    },
                }
            ],
        )

    def test_run_loop_executes_function_call_and_appends_function_call_output(self):
        agent = self.make_agent()
        agent.client = FakeClient(
            [
                make_response(
                    output=[
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "Need evidence."}],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "search",
                            "arguments": '{"query":"alpha"}',
                        },
                    ]
                ),
                make_response(
                    output=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Final answer."}],
                        }
                    ],
                    output_text="Final answer.",
                    response_id="resp_2",
                ),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 2)
        self.assertEqual(agent.state.tool_rounds, 1)
        self.assertTrue(
            any(
                msg.get("type") == "function_call_output"
                and msg.get("call_id") == "call_1"
                and msg.get("output") == "search:alpha"
                for msg in agent.messages
            )
        )
        self.assertEqual(
            agent.trajectory[0].tool_calls[0].result,
            "search:alpha",
        )

    def test_finalize_can_continue_tool_loop_after_stop_prompt(self):
        agent = self.make_agent(max_steps=1)
        agent.client = FakeClient(
            [
                make_response(
                    output=[
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "search",
                            "arguments": '{"query":"alpha"}',
                        }
                    ]
                ),
                make_response(
                    output=[
                        {
                            "type": "function_call",
                            "id": "fc_2",
                            "call_id": "call_2",
                            "name": "search",
                            "arguments": '{"query":"beta"}',
                        }
                    ],
                    response_id="resp_2",
                ),
                make_response(
                    output=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Final answer."}],
                        }
                    ],
                    output_text="Final answer.",
                    response_id="resp_3",
                ),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 3)
        self.assertEqual(agent.state.tool_rounds, 2)
        self.assertTrue(
            any(
                msg.get("type") == "message"
                and msg.get("role") == "user"
                and "maximum number of steps" in msg.get("content", "")
                for msg in agent.messages
            )
        )


if __name__ == "__main__":
    unittest.main()
