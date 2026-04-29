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
from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.base_tool import BaseTool


class DummyTool(BaseTool):
    def __init__(self, name: str = "search"):
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
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class QwenAgentXmlToolTests(unittest.TestCase):
    def make_agent(
        self,
        *,
        llm_config: LLMConfig | None = None,
        max_steps: int = 3,
    ) -> QwenAgent:
        return QwenAgent(
            model="fake-model",
            api_key="fake-key",
            base_url="http://localhost",
            tools=[DummyTool()],
            llm_config=llm_config or LLMConfig(stream=False),
            max_steps=max_steps,
        )

    def test_system_prompt_inlines_xml_tool_contract(self):
        agent = self.make_agent()

        self.assertIn("# Tools", agent.system_prompt)
        self.assertIn("<tools>", agent.system_prompt)
        self.assertIn('"type": "function"', agent.system_prompt)
        self.assertIn('"function": {', agent.system_prompt)
        self.assertIn('"name": "search"', agent.system_prompt)
        self.assertIn("<tool_call>", agent.system_prompt)
        self.assertIn("<function=example_function_name>", agent.system_prompt)
        self.assertNotIn("{tool_des}", agent.system_prompt)

    def test_call_llm_sends_no_native_tools(self):
        agent = self.make_agent(
            llm_config=LLMConfig(
                max_tokens=256,
                temperature=0.2,
                top_p=0.8,
                tool_choice="auto",
                stream=False,
            )
        )
        agent.client = FakeClient([make_response("done")])
        agent.messages = [{"role": "user", "content": "hello"}]

        agent._call_llm()

        payload = agent.client.chat.completions.calls[0]
        self.assertEqual(payload["model"], "fake-model")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("stream", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("<tools>", payload["messages"][0]["content"])

    def test_second_llm_call_replays_xml_history_as_plain_messages(self):
        agent = self.make_agent(max_steps=3)
        first_assistant = (
            "<think>\nNeed another search.\n</think>\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\n"
            '["alpha"]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        agent.client = FakeClient(
            [
                make_response(first_assistant),
                make_response("<think>Enough evidence.</think>\nFinal answer."),
            ]
        )

        agent.run_verbose("hello")

        second_payload = agent.client.chat.completions.calls[1]
        self.assertNotIn("tools", second_payload)
        self.assertNotIn("tool_choice", second_payload)
        self.assertEqual(second_payload["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(
            second_payload["messages"][2],
            {"role": "assistant", "content": first_assistant},
        )
        self.assertEqual(second_payload["messages"][3]["role"], "user")
        self.assertTrue(
            second_payload["messages"][3]["content"].startswith("<tool_response>\n")
        )
        self.assertTrue(
            second_payload["messages"][3]["content"].endswith("\n</tool_response>\n")
        )

    def test_parse_response_extracts_think_xml_tool_and_no_final_content(self):
        agent = self.make_agent()
        raw = (
            "<think>Need evidence.</think>\n\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\n"
            '["alpha", "beta"]\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        parsed = agent._parse_response(raw)

        self.assertEqual(parsed["reasoning_content"], "Need evidence.")
        self.assertEqual(parsed["final_content"], "")
        self.assertFalse(parsed.get("retryable"))
        self.assertEqual(
            parsed["tool_calls"],
            [
                {
                    "function": {
                        "name": "search",
                        "arguments": {"query": ["alpha", "beta"]},
                    }
                }
            ],
        )
        self.assertEqual(parsed["assistant_message"], {"role": "assistant", "content": raw})

    def test_parse_response_supports_no_open_think_tag(self):
        agent = self.make_agent()
        raw = "Need evidence before searching.\n</think>\n\nFinal answer."

        parsed = agent._parse_response(raw)

        self.assertEqual(parsed["reasoning_content"], "Need evidence before searching.")
        self.assertEqual(parsed["final_content"], "Final answer.")

    def test_parse_response_marks_malformed_tool_markup_retryable(self):
        agent = self.make_agent()
        raw = (
            "I should search for the source first.\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        parsed = agent._parse_response(raw)

        self.assertEqual(parsed["tool_calls"], [])
        self.assertTrue(parsed["retryable"])
        self.assertEqual(parsed["retry_reason"], "malformed_tool_call_xml")
        self.assertEqual(parsed["final_content"], "")

    def test_treats_formatted_answer_in_reasoning_as_final_content(self):
        agent = self.make_agent()
        raw = (
            "<think>\n"
            "Explanation: Evidence supports the answer. [1]\n\n"
            "Exact Answer: Amherst College\n\n"
            "Confidence: 100%\n"
            "</think>"
        )

        parsed = agent._parse_response(raw)

        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(
            parsed["final_content"],
            (
                "Explanation: Evidence supports the answer. [1]\n\n"
                "Exact Answer: Amherst College\n\n"
                "Confidence: 100%"
            ),
        )

    def test_unclosed_thinking_with_thinking_enabled_is_retryable(self):
        agent = self.make_agent(llm_config=LLMConfig(stream=False, enable_thinking=True))
        raw = "I am still thinking. " * 700

        parsed = agent._parse_response(raw)

        self.assertTrue(parsed["retryable"])
        self.assertEqual(parsed["retry_reason"], "unclosed_qwen_thinking")
        self.assertEqual(parsed["final_content"], "")

    def test_run_loop_executes_xml_tool_and_appends_tool_response_user_message(self):
        agent = self.make_agent(max_steps=3)
        agent.client = FakeClient(
            [
                make_response(
                    "<think>Need another search.</think>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    '["alpha"]\n'
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response("<think>Enough evidence.</think>\nFinal answer."),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        self.assertEqual(agent.state.step, 2)
        self.assertEqual(agent.state.tool_rounds, 1)
        self.assertEqual(agent.trajectory[0].reasoning, "Need another search.")
        self.assertTrue(
            any(
                msg.get("role") == "user"
                and msg.get("content", "").startswith("<tool_response>")
                and "search:['alpha']" in msg.get("content", "")
                for msg in agent.messages
            )
        )

    def test_run_loop_groups_multiple_tool_responses_in_one_user_message(self):
        agent = self.make_agent(max_steps=3)
        agent.client = FakeClient(
            [
                make_response(
                    "<think>Need two searches.</think>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    '["alpha"]\n'
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>\n"
                    "<tool_call>\n"
                    "<function=search>\n"
                    "<parameter=query>\n"
                    '["beta"]\n'
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                make_response("Final answer."),
            ]
        )

        result = agent.run_verbose("hello")

        self.assertEqual(result["final_content"], "Final answer.")
        second_payload = agent.client.chat.completions.calls[1]
        self.assertEqual(len(second_payload["messages"]), 4)
        tool_message = second_payload["messages"][3]
        self.assertEqual(tool_message["role"], "user")
        self.assertEqual(tool_message["content"].count("<tool_response>"), 2)
        self.assertIn("search:['alpha']", tool_message["content"])
        self.assertIn("search:['beta']", tool_message["content"])
        self.assertFalse(
            any(
                msg.get("role") == "user"
                and msg is not tool_message
                and msg.get("content", "").startswith("<tool_response>")
                for msg in second_payload["messages"]
            )
        )

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
                    '["alpha"]\n'
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
        self.assertEqual(agent.messages[0], {"role": "user", "content": "hello"})
        self.assertIn("<think>Search first.</think>", agent.messages[1]["content"])
        self.assertNotIn("I should search for the source first.", agent.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
