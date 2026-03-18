import unittest

from insikt.analysis import extract_keywords
from insikt.exports import export_markdown, export_text


class AnalysisExportTests(unittest.TestCase):
    def test_extract_keywords_returns_items(self):
        keywords = extract_keywords("journalism corruption minister budget reform transparency", top_n=3)
        self.assertTrue(len(keywords) > 0)

    def test_export_helpers_return_bytes(self):
        self.assertIsInstance(export_text("hej"), bytes)
        self.assertIsInstance(export_markdown("hej"), bytes)


if __name__ == "__main__":
    unittest.main()
