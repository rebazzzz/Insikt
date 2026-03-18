import unittest

from langchain_core.documents import Document

from insikt.analysis import compare_sources, extract_claim_check_items, extract_keywords, extract_quote_candidates
from insikt.common import safe_html_fragment
from insikt.exports import build_export_sections, export_markdown, export_text


class AnalysisExportTests(unittest.TestCase):
    def test_extract_keywords_returns_items(self):
        keywords = extract_keywords("journalism corruption minister budget reform transparency", top_n=3)
        self.assertTrue(len(keywords) > 0)

    def test_export_helpers_return_bytes(self):
        self.assertIsInstance(export_text("hej"), bytes)
        self.assertIsInstance(export_markdown("hej"), bytes)

    def test_safe_export_adds_endnotes_and_appendix(self):
        package = build_export_sections("Claim [Source: report.pdf, page 2]", "en")
        self.assertIn("Endnotes", package["full_text"])
        self.assertIn("Source appendix", package["full_text"])
        self.assertIn("[1] report.pdf, page 2", package["full_text"])

    def test_plain_text_export_keeps_original_text_when_safe_mode_is_off(self):
        text = "Claim [Source: report.pdf, page 2]"
        exported = export_text(text, "en", safe_mode=False).decode("utf-8")
        self.assertEqual(exported, text)

    def test_quote_extraction_returns_source_and_page(self):
        docs = [Document(page_content='He said, "This is a very important statement for the investigation." Later text.', metadata={"source": "notes.pdf", "page": "4"})]
        quotes = extract_quote_candidates(docs, max_quotes=5)
        self.assertEqual(quotes[0]["source"], "notes.pdf")
        self.assertEqual(str(quotes[0]["page"]), "4")
        self.assertIn("important statement", quotes[0]["quote"])

    def test_claim_checker_flags_uncited_numeric_sentence(self):
        items = extract_claim_check_items("The ministry spent 4.2 million euros on the project.")
        flagged = [item for item in items if item["needs_review"]]
        self.assertTrue(flagged)
        self.assertIn("number_without_citation", flagged[0]["reasons"])

    def test_compare_sources_returns_matching_excerpts(self):
        docs = [
            Document(page_content="The minister approved the 2024 budget after a late meeting.", metadata={"source": "a.pdf", "page": "1"}),
            Document(page_content="Sports coverage and weather only.", metadata={"source": "b.pdf", "page": "2"}),
        ]
        results = compare_sources(docs, "minister budget 2024")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "a.pdf")
        self.assertIn("2024 budget", results[0]["excerpt"])

    def test_safe_html_fragment_escapes_markup(self):
        escaped = safe_html_fragment("<script>alert(1)</script>\nline")
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;<br>line", escaped)


if __name__ == "__main__":
    unittest.main()
