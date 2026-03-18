import shutil
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from insikt.common import docs_to_records, records_to_docs
from insikt.pipeline import get_cache_stats, pdf_needs_ocr
from insikt.session_store import delete_slot, list_save_slots, save_slot


class DummyVectorstore:
    def save_local(self, path: str):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_text("fake", encoding="utf-8")
        (folder / "index.pkl").write_text("fake", encoding="utf-8")


class PipelineSessionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_doc_record_roundtrip(self):
        docs = [Document(page_content="hello", metadata={"source": "a.pdf", "page": "1"})]
        self.assertEqual(records_to_docs(docs_to_records(docs))[0].page_content, "hello")

    def test_save_slots_can_be_listed(self):
        docs = [Document(page_content="hello", metadata={"source": "a.pdf", "page": "1"})]
        slot_id = save_slot(
            self.tempdir,
            "Case A",
            docs,
            docs,
            [{"role": "user", "content": "hi"}],
            "summary",
            "sv",
            DummyVectorstore(),
            "abc",
            case_folder="Investigations",
            tags=["fraud", "budget"],
            case_board={"notes": "Lead note", "people": ["Minister A"]},
        )
        slots = list_save_slots(self.tempdir)
        self.assertEqual(slot_id, "Case-A")
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["case_folder"], "Investigations")
        self.assertEqual(slots[0]["tags"], ["fraud", "budget"])
        self.assertTrue(slots[0]["preview_text"])
        self.assertEqual(slots[0]["preview_source"], "a.pdf")
        from insikt.session_store import load_slot

        payload = load_slot(self.tempdir, slot_id, None)
        self.assertEqual(payload["case_board"]["notes"], "Lead note")
        delete_slot(self.tempdir, slot_id)
        self.assertEqual(list_save_slots(self.tempdir), [])

    def test_cache_stats_report_file_and_bundle_counts(self):
        (self.tempdir / "abc123").mkdir(parents=True, exist_ok=True)
        (self.tempdir / "file_pages" / "file1").mkdir(parents=True, exist_ok=True)
        stats = get_cache_stats(self.tempdir)
        self.assertEqual(stats["bundle_caches"], 1)
        self.assertEqual(stats["file_caches"], 1)

    def test_pdf_needs_ocr_when_pages_have_no_meaningful_text(self):
        docs = [Document(page_content="   ", metadata={"source": "scan.pdf", "page": "1"})]
        self.assertTrue(pdf_needs_ocr(docs))

    def test_pdf_does_not_need_ocr_when_text_is_present(self):
        docs = [Document(page_content="This page contains enough extracted PDF text to skip OCR.", metadata={"source": "text.pdf", "page": "1"})]
        self.assertFalse(pdf_needs_ocr(docs))


if __name__ == "__main__":
    unittest.main()
