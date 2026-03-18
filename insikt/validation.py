from __future__ import annotations

import importlib.util
import subprocess

import torch


def get_installed_ollama_models() -> list[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0].strip())
        return models
    except Exception:
        return []


def resolve_installed_ollama_model(requested_model: str, installed_models: list[str]) -> str | None:
    if not installed_models:
        return requested_model
    if requested_model in installed_models:
        return requested_model
    if ":" not in requested_model:
        latest_variant = f"{requested_model}:latest"
        if latest_variant in installed_models:
            return latest_variant
    base_name = requested_model.split(":", 1)[0]
    for model_name in installed_models:
        if model_name == base_name or model_name.startswith(base_name + ":"):
            return model_name
    return None


def run_startup_checks(llm_model: str, embedding_model: str) -> list[dict]:
    checks = []
    required_modules = [
        "streamlit",
        "torch",
        "transformers",
        "langchain_community",
        "langchain_huggingface",
        "langchain_ollama",
        "faiss",
        "pypdf",
        "docx",
        "fpdf",
        "yake",
        "sentence_transformers",
    ]
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    checks.append({"name": "Python dependencies", "status": "ok" if not missing else "warning", "message": "All required packages detected." if not missing else f"Missing packages: {', '.join(missing)}"})
    checks.append({"name": "GPU readiness", "status": "ok" if torch.cuda.is_available() else "info", "message": "CUDA GPU available." if torch.cuda.is_available() else "CUDA GPU not detected. CPU mode will be used when needed."})
    try:
        installed_models = get_installed_ollama_models()
        output = "\n".join(installed_models)
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5, check=False)
        has_ollama = result.returncode == 0
        checks.append({"name": "Ollama", "status": "ok" if has_ollama else "warning", "message": "Ollama responded successfully." if has_ollama else "Ollama did not respond. Start it before using chat or summaries."})
        resolved_model = resolve_installed_ollama_model(llm_model, installed_models)
        if resolved_model:
            msg = f"Model '{llm_model}' detected as '{resolved_model}'."
            status = "ok"
        elif installed_models:
            msg = f"Model '{llm_model}' was not found. Installed models: {', '.join(installed_models[:5])}."
            status = "warning"
        else:
            msg = f"Model '{llm_model}' could not be verified."
            status = "warning"
        checks.append({"name": "Selected LLM", "status": status, "message": msg})
    except Exception:
        checks.append({"name": "Ollama", "status": "warning", "message": "Could not run 'ollama list'. Verify that Ollama is installed and on PATH."})
        checks.append({"name": "Selected LLM", "status": "info", "message": f"Could not verify model '{llm_model}' because Ollama was unavailable."})
    checks.append({"name": "Embedding profile", "status": "ok", "message": f"Embedding profile '{embedding_model}' is configured."})
    return checks
