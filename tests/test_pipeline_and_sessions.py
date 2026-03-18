import shutil
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from insikt.common import docs_to_records, records_to_docs
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
        slot_id = save_slot(self.tempdir, "Case A", docs, docs, [{"role": "user", "content": "hi"}], "summary", "sv", DummyVectorstore(), "abc")
        slots = list_save_slots(self.tempdir)
        self.assertEqual(slot_id, "Case-A")
        self.assertEqual(len(slots), 1)
        delete_slot(self.tempdir, slot_id)
        self.assertEqual(list_save_slots(self.tempdir), [])


if __name__ == "__main__":
    unittest.main()
