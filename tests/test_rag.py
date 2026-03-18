import unittest

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from insikt.rag import assess_answer_confidence, create_chat_prompt, extract_citations_from_response, rerank_documents, verify_citations


class RagTests(unittest.TestCase):
    def test_prompt_contains_context(self):
        docs = [Document(page_content="Budget fraud on page one", metadata={"source": "report.pdf", "page": "1"})]
        prompt = create_chat_prompt([HumanMessage(content="What happened?")], docs, "What happened?", "en")
        self.assertIn("report.pdf", prompt)

    def test_extract_citations(self):
        citations = extract_citations_from_response("Claim [Source: report.pdf, page 2]")
        self.assertEqual(citations[0]["page"], "2")

    def test_verify_citations_passes_known_source(self):
        docs = [Document(page_content="x", metadata={"source": "report.pdf", "page": "2"})]
        _, issues = verify_citations("Claim [Source: report.pdf, page 2]", docs, "en")
        self.assertEqual(issues, [])

    def test_rerank_prioritizes_overlap(self):
        docs = [
            Document(page_content="This is about sports only", metadata={"source": "a", "page": "1"}),
            Document(page_content="Corruption scandal involving the budget committee", metadata={"source": "b", "page": "1"}),
        ]
        ranked = rerank_documents("budget corruption", docs, limit=1)
        self.assertEqual(ranked[0].metadata["source"], "b")

    def test_confidence_well_supported_for_multiple_verified_citations(self):
        docs = [
            Document(page_content="Budget fraud details", metadata={"source": "report.pdf", "page": "2"}),
            Document(page_content="Witness statement", metadata={"source": "notes.pdf", "page": "5"}),
        ]
        result = assess_answer_confidence(
            "Claim [Source: report.pdf, page 2] and corroboration [Source: notes.pdf, page 5]",
            docs,
            [],
            "en",
        )
        self.assertEqual(result["level"], "well_supported")

    def test_confidence_needs_review_without_citations(self):
        docs = [Document(page_content="Budget fraud details", metadata={"source": "report.pdf", "page": "2"})]
        result = assess_answer_confidence("Claim without support", docs, ["missing_citations"], "en")
        self.assertEqual(result["level"], "needs_review")


if __name__ == "__main__":
    unittest.main()
