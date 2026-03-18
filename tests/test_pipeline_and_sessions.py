import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from langchain_core.documents import Document

from insikt.common import docs_to_records, records_to_docs
from insikt.pipeline import get_cache_stats, load_single_pdf, ocr_stack_available, pdf_needs_ocr
from insikt.session_store import delete_slot, list_save_slots, save_slot


class DummyVectorstore:
    def save_local(self, path: str):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_text("fake", encoding="utf-8")
        (folder / "index.pkl").write_text("fake", encoding="utf-8")


class UploadedFileDouble:
    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self):
        return self._bytes


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

    def test_scanned_pdf_uses_ocr_when_stack_is_available(self):
        if not ocr_stack_available():
            self.skipTest("OCR stack is not available in this environment")
        pdf_path = self.tempdir / "scanned.pdf"
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 88)
        image = Image.new("RGB", (2200, 1400), "white")
        draw = ImageDraw.Draw(image)
        draw.multiline_text(
            (140, 180),
            "INSIKT OCR TEST\nBudget notes 2026\nAnna Berg confirmed 42 invoices.",
            fill="black",
            font=font,
            spacing=28,
        )
        image.save(pdf_path, "PDF", resolution=300.0)

        docs = load_single_pdf(UploadedFileDouble(pdf_path))

        self.assertTrue(docs)
        self.assertTrue(docs[0].metadata.get("ocr"))
        self.assertIn("budget", docs[0].page_content.lower())


if __name__ == "__main__":
    unittest.main()
