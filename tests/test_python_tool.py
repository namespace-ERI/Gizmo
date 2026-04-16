import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Gizmo.tools import PythonTool


class PythonToolTests(unittest.TestCase):
    def test_schema_uses_expected_function_name(self):
        tool = PythonTool()

        schema = tool.to_schema()

        self.assertEqual(schema["function"]["name"], "code_interpreter")
        self.assertIn("code", schema["function"]["parameters"]["properties"])
        tool.shutdown()

    def test_execute_captures_stdout(self):
        tool = PythonTool()

        result = tool.execute("x = 5\nprint(x * 2)")

        self.assertIn("stdout:", result)
        self.assertIn("10", result)
        tool.shutdown()

    def test_execute_persists_state_across_calls(self):
        tool = PythonTool()

        tool.execute("x = 5")
        result = tool.execute("print(x + 2)")

        self.assertIn("stdout:", result)
        self.assertIn("7", result)
        tool.shutdown()

    def test_execute_blocks_unsafe_code(self):
        tool = PythonTool()

        result = tool.execute("import os\nprint('hello')")

        self.assertIn("error:", result)
        self.assertIn("unsafe code", result.lower())
        tool.shutdown()


if __name__ == "__main__":
    unittest.main()
