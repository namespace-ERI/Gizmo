import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Gizmo.tools import GoogleScholarTool


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class GoogleScholarToolTests(unittest.TestCase):
    def test_schema_uses_expected_function_name(self):
        tool = GoogleScholarTool(api_key="fake-key")

        schema = tool.to_schema()

        self.assertEqual(schema["function"]["name"], "google_scholar")
        self.assertIn("query", schema["function"]["parameters"]["properties"])

    @patch("Gizmo.tools.google_scholar_tool.requests.post")
    def test_execute_single_formats_results(self, mock_post):
        mock_post.return_value = _FakeResponse(
            200,
            {
                "organic": [
                    {
                        "title": "Attention Is All You Need",
                        "pdfUrl": "https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
                        "publicationInfo": "A Vaswani - NeurIPS, 2017",
                        "year": 2017,
                        "citedBy": 100000,
                        "snippet": "Transformer paper.",
                    }
                ]
            },
        )
        tool = GoogleScholarTool(api_key="fake-key")

        result = tool.execute(["transformer paper"])

        self.assertIn("A Google scholar for 'transformer paper' found 1 results", result)
        self.assertIn("Attention Is All You Need", result)
        self.assertIn("Date published: 2017", result)
        self.assertIn("publicationInfo: A Vaswani - NeurIPS, 2017", result)
        self.assertIn("citedBy: 100000", result)

    @patch("Gizmo.tools.google_scholar_tool.requests.post")
    def test_execute_batch_joins_results_in_query_order(self, mock_post):
        mock_post.side_effect = [
            _FakeResponse(200, {"organic": [{"title": "Paper A", "snippet": "alpha"}]}),
            _FakeResponse(200, {"organic": [{"title": "Paper B", "snippet": "beta"}]}),
        ]
        tool = GoogleScholarTool(api_key="fake-key", max_workers=2)

        result = tool.execute(["query a", "query b"])

        self.assertIn("query a", result)
        self.assertIn("query b", result)
        self.assertIn("======= ", result.replace("\n", " ")[:2000] or "=======")
        self.assertLess(result.index("query a"), result.index("query b"))

    @patch("Gizmo.tools.google_scholar_tool.requests.post")
    def test_execute_surfaces_auth_errors(self, mock_post):
        mock_post.return_value = _FakeResponse(
            200,
            {"error": "Invalid or expired token"},
        )
        tool = GoogleScholarTool(api_key="bad-key")

        result = tool.execute(["query a"])

        self.assertIn("Scholar Error: PermissionError:", result)
        self.assertIn("Invalid or expired token", result)


if __name__ == "__main__":
    unittest.main()
