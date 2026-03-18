from __future__ import annotations

import importlib.util
import subprocess

import torch


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
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5, check=False)
        output = result.stdout + "\n" + result.stderr
        has_ollama = result.returncode == 0
        checks.append({"name": "Ollama", "status": "ok" if has_ollama else "warning", "message": "Ollama responded successfully." if has_ollama else "Ollama did not respond. Start it before using chat or summaries."})
        checks.append({"name": "Selected LLM", "status": "ok" if llm_model in output else "warning", "message": f"Model '{llm_model}' detected." if llm_model in output else f"Model '{llm_model}' was not found in 'ollama list'."})
    except Exception:
        checks.append({"name": "Ollama", "status": "warning", "message": "Could not run 'ollama list'. Verify that Ollama is installed and on PATH."})
        checks.append({"name": "Selected LLM", "status": "info", "message": f"Could not verify model '{llm_model}' because Ollama was unavailable."})
    checks.append({"name": "Embedding profile", "status": "ok", "message": f"Embedding profile '{embedding_model}' is configured."})
    return checks
