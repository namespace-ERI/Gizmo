import unittest

from Gizmo.tools.search_tool import SearchTool


class SearchToolQueryRepairTests(unittest.TestCase):
    def test_coerce_queries_keeps_valid_json_array(self):
        self.assertEqual(
            SearchTool._coerce_queries('["alpha", "beta"]'),
            ["alpha", "beta"],
        )

    def test_repairs_missing_brackets_quoted_batch(self):
        self.assertEqual(
            SearchTool._coerce_queries('"alpha query", "beta query", "gamma query"'),
            ["alpha query", "beta query", "gamma query"],
        )

    def test_repairs_missing_opening_quote_batch_from_xml(self):
        value = (
            "MMA featherweight 14 significant strikes 83 attempted 16.87% "
            'loser takedown 0 from 4 attempts", '
            '"MMA fighter nickname swordsman swordsman synonym fighter", '
            '"UFC featherweight bout significant strikes percentage 16.87"]'
        )

        self.assertEqual(
            SearchTool._coerce_queries(value),
            [
                (
                    "MMA featherweight 14 significant strikes 83 attempted "
                    "16.87% loser takedown 0 from 4 attempts"
                ),
                "MMA fighter nickname swordsman swordsman synonym fighter",
                "UFC featherweight bout significant strikes percentage 16.87",
            ],
        )

    def test_repairs_batch_while_preserving_phrase_quotes(self):
        value = (
            '"album for fun" 2022 article English artist debut 2001 2005", '
            '"I do not feel threatened except when another artist like me surfaces" interview", '
            '"English artist Roman numeral I II albums 2000s"]'
        )

        self.assertEqual(
            SearchTool._coerce_queries(value),
            [
                '"album for fun" 2022 article English artist debut 2001 2005',
                (
                    '"I do not feel threatened except when another artist like me '
                    'surfaces" interview'
                ),
                "English artist Roman numeral I II albums 2000s",
            ],
        )

    def test_cleans_broken_single_query_array_shell_without_removing_phrase_quotes(self):
        value = '["Artificial Neural Networks" paper August 1991 August 1993 August 1995"]'

        self.assertEqual(
            SearchTool._coerce_queries(value),
            ['"Artificial Neural Networks" paper August 1991 August 1993 August 1995'],
        )

    def test_repairs_newline_query_batch(self):
        value = """
        1. artificial neural networks Academy of Management August 1991
        2. artificial neural networks Academy of Management August 1993
        3. artificial neural networks Academy of Management August 1995
        """

        self.assertEqual(
            SearchTool._coerce_queries(value),
            [
                "artificial neural networks Academy of Management August 1991",
                "artificial neural networks Academy of Management August 1993",
                "artificial neural networks Academy of Management August 1995",
            ],
        )


if __name__ == "__main__":
    unittest.main()
