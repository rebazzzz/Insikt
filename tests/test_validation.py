import unittest
from unittest.mock import patch

from insikt.validation import (
    choose_model_variant,
    get_missing_python_packages,
    get_model_recommendations,
    get_tesseract_install_hint,
    install_missing_python_packages,
    resolve_tesseract_command,
    resolve_installed_ollama_model,
)


class ValidationTests(unittest.TestCase):
    def test_resolve_installed_model_prefers_latest_variant(self):
        resolved = resolve_installed_ollama_model("llama3.2", ["llama3.2:latest", "mistral:7b"])
        self.assertEqual(resolved, "llama3.2:latest")

    def test_choose_model_variant_falls_back_to_first_installed_model(self):
        chosen = choose_model_variant(["llama3.2:latest"], ["custom-model:latest"], ["llama3.2:latest"])
        self.assertEqual(chosen, "custom-model:latest")

    def test_recommendations_for_entry_machine_prefer_small_models(self):
        profile = {"gpu_available": False, "vram_gb": 0.0, "ram_gb": 8.0, "cpu_count": 4}
        presets = get_model_recommendations([], profile, ["llama3.2:1b", "llama3.2:3b", "llama3.2:latest"])
        preset_map = {preset["key"]: preset for preset in presets}
        self.assertEqual(preset_map["fast"]["llm_model"], "llama3.2:1b")
        self.assertEqual(preset_map["balanced"]["llm_model"], "llama3.2:1b")
        self.assertEqual(preset_map["best"]["llm_model"], "llama3.2:3b")

    def test_recommendations_use_installed_models_when_available(self):
        profile = {"gpu_available": True, "vram_gb": 12.0, "ram_gb": 32.0, "cpu_count": 12}
        presets = get_model_recommendations(["mistral:7b"], profile, ["llama3.2:1b", "llama3.2:3b", "llama3.2:latest", "mistral:7b"])
        preset_map = {preset["key"]: preset for preset in presets}
        self.assertEqual(preset_map["balanced"]["llm_model"], "mistral:7b")
        self.assertTrue(preset_map["balanced"]["installed"])

    @patch("insikt.validation.importlib.util.find_spec")
    def test_missing_python_packages_map_import_names_to_pip_names(self, mock_find_spec):
        mock_find_spec.side_effect = lambda name: None if name in {"pytesseract", "PIL"} else object()
        missing = get_missing_python_packages(["pytesseract", "PIL", "streamlit"])
        self.assertEqual(missing, [{"module": "pytesseract", "package": "pytesseract"}, {"module": "PIL", "package": "Pillow"}])

    @patch("insikt.validation.install_python_packages")
    @patch("insikt.validation.get_missing_python_packages")
    def test_install_missing_python_packages_uses_unique_pip_names(self, mock_missing, mock_install):
        mock_missing.return_value = [
            {"module": "PIL", "package": "Pillow"},
            {"module": "custom", "package": "Pillow"},
            {"module": "ocr", "package": "pytesseract"},
        ]
        install_missing_python_packages()
        mock_install.assert_called_once_with(["Pillow", "Pillow", "pytesseract"])

    def test_windows_tesseract_hint_mentions_winget(self):
        hint = get_tesseract_install_hint()
        self.assertTrue("Tesseract" in hint)

    @patch("insikt.validation.os.path.exists")
    @patch("insikt.validation.which")
    def test_resolve_tesseract_command_checks_common_windows_locations(self, mock_which, mock_exists):
        mock_which.return_value = None
        mock_exists.side_effect = lambda path: str(path).endswith("Tesseract-OCR\\tesseract.exe")
        resolved = resolve_tesseract_command()
        self.assertTrue(resolved.lower().endswith("tesseract.exe"))


if __name__ == "__main__":
    unittest.main()
