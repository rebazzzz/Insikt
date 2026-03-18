import unittest

from insikt.validation import choose_model_variant, get_model_recommendations, resolve_installed_ollama_model


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


if __name__ == "__main__":
    unittest.main()
