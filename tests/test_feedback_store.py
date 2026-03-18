import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from insikt.feedback_store import build_feedback_bundle, list_issue_reports, save_issue_report


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_save_issue_report_creates_unique_files_without_overwriting(self):
        payload = {
            "title": "Chat answer looked wrong",
            "reporter_name": "Tester",
            "severity": "High",
            "area": "Chat",
            "what_happened": "The answer mixed up two people.",
            "expected": "Separate the people correctly.",
            "steps": "Upload docs and ask about the minister.",
            "work_context": "Real workflow",
            "app_version_label": "0.3.0-early",
            "app_context": {"sources": ["a.pdf"]},
        }
        first = save_issue_report(self.tempdir, payload)
        second = save_issue_report(self.tempdir, payload)

        self.assertNotEqual(first["report_id"], second["report_id"])
        reports = list_issue_reports(self.tempdir)
        self.assertEqual(len(reports), 2)

    def test_build_feedback_bundle_contains_manifest_and_report_files(self):
        payload = {
            "title": "Export wording issue",
            "reporter_name": "Tester",
            "severity": "Medium",
            "area": "Export",
            "what_happened": "The export note was confusing.",
            "expected": "Clearer export wording.",
            "steps": "Generate summary then export.",
            "work_context": "Dry run",
            "app_version_label": "0.3.0-early",
            "app_context": {"lang": "en"},
        }
        report = save_issue_report(self.tempdir, payload)
        bundle_bytes = build_feedback_bundle(self.tempdir, [report["report_id"]])

        with zipfile.ZipFile(Path(self.tempdir / "bundle.zip"), "w") as _:
            pass
        archive_path = self.tempdir / "bundle_check.zip"
        archive_path.write_bytes(bundle_bytes)
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            self.assertIn("manifest.json", names)
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest[0]["report_id"], report["report_id"])
            self.assertTrue(any(name.endswith(".json") for name in names))
            self.assertTrue(any(name.endswith(".md") for name in names))


if __name__ == "__main__":
    unittest.main()
