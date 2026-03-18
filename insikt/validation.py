from __future__ import annotations

import importlib.util
import os
import subprocess
from shutil import which

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


def ollama_cli_available() -> bool:
    return which("ollama") is not None


def tesseract_cli_available() -> bool:
    return which("tesseract") is not None


def get_system_memory_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024 ** 3), 1)
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            return round((page_size * page_count) / (1024 ** 3), 1)
    except Exception:
        return 0.0
    return 0.0


def get_system_profile() -> dict:
    cpu_count = os.cpu_count() or 0
    ram_gb = get_system_memory_gb()
    gpu_available = torch.cuda.is_available()
    gpu_name = ""
    vram_gb = 0.0
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            vram_gb = round(properties.total_memory / (1024 ** 3), 1)
        except Exception:
            gpu_name = "CUDA GPU"
    if gpu_available and vram_gb >= 10:
        tier = "high"
    elif gpu_available or ram_gb >= 16 or cpu_count >= 8:
        tier = "medium"
    else:
        tier = "entry"
    return {
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "tier": tier,
        "ollama_cli": ollama_cli_available(),
    }


def choose_model_variant(preferred_models: list[str], installed_models: list[str], known_models: list[str]) -> str:
    if installed_models:
        for candidate in preferred_models:
            resolved = resolve_installed_ollama_model(candidate, installed_models)
            if resolved:
                return resolved
        return installed_models[0]
    for candidate in preferred_models:
        if candidate in known_models:
            return candidate
    return preferred_models[0]


def get_model_recommendations(
    installed_models: list[str],
    profile: dict | None = None,
    known_models: list[str] | None = None,
) -> list[dict]:
    profile = profile or get_system_profile()
    known_models = known_models or []
    ram_gb = profile.get("ram_gb", 0.0)
    cpu_count = profile.get("cpu_count", 0)
    gpu_available = profile.get("gpu_available", False)
    vram_gb = profile.get("vram_gb", 0.0)

    if gpu_available and vram_gb >= 10:
        balanced_targets = ["llama3.2:3b", "llama3.2:latest"]
        best_targets = ["llama3.2:latest", "mistral:7b"]
    elif ram_gb >= 24 or cpu_count >= 12:
        balanced_targets = ["llama3.2:3b", "llama3.2:latest"]
        best_targets = ["llama3.2:latest", "mistral:7b"]
    elif ram_gb >= 16 or cpu_count >= 8:
        balanced_targets = ["llama3.2:3b", "llama3.2:latest"]
        best_targets = ["llama3.2:3b", "llama3.2:latest"]
    else:
        balanced_targets = ["llama3.2:1b", "llama3.2:3b"]
        best_targets = ["llama3.2:3b", "llama3.2:1b"]

    presets = [
        {
            "key": "fast",
            "label": "Fast",
            "llm_model": choose_model_variant(["llama3.2:1b", "llama3.2:3b"], installed_models, known_models),
            "embedding_model": "bge-small",
            "summary": "Lowest memory use and quickest responses.",
        },
        {
            "key": "balanced",
            "label": "Balanced",
            "llm_model": choose_model_variant(balanced_targets, installed_models, known_models),
            "embedding_model": "bge-base",
            "summary": "Best everyday choice for reporting and drafting.",
        },
        {
            "key": "best",
            "label": "Best quality",
            "llm_model": choose_model_variant(best_targets, installed_models, known_models),
            "embedding_model": "bge-base",
            "summary": "Highest quality this computer is likely to handle well.",
        },
    ]
    for preset in presets:
        preset["installed"] = preset["llm_model"] in installed_models if installed_models else False
    return presets


def run_startup_checks(llm_model: str, embedding_model: str) -> list[dict]:
    checks = []
    profile = get_system_profile()
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
        "pytesseract",
        "pypdfium2",
        "PIL",
        "yake",
        "sentence_transformers",
    ]
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    checks.append({"name": "Python dependencies", "status": "ok" if not missing else "warning", "message": "All required packages detected." if not missing else f"Missing packages: {', '.join(missing)}"})
    if profile["gpu_available"]:
        gpu_message = f"CUDA GPU available: {profile['gpu_name']} ({profile['vram_gb']} GB VRAM)."
        gpu_status = "ok"
    else:
        gpu_message = f"No CUDA GPU detected. CPU mode will be used when needed. System RAM: {profile['ram_gb']} GB."
        gpu_status = "info"
    checks.append({"name": "GPU readiness", "status": gpu_status, "message": gpu_message})
    try:
        installed_models = get_installed_ollama_models()
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
        cli_hint = "Verify that Ollama is installed and on PATH." if ollama_cli_available() else "Ollama CLI was not found on PATH."
        checks.append({"name": "Ollama", "status": "warning", "message": f"Could not run 'ollama list'. {cli_hint}"})
        checks.append({"name": "Selected LLM", "status": "info", "message": f"Could not verify model '{llm_model}' because Ollama was unavailable."})
    checks.append({"name": "Embedding profile", "status": "ok", "message": f"Embedding profile '{embedding_model}' is configured."})
    ocr_modules_ready = all(importlib.util.find_spec(name) is not None for name in ["pytesseract", "pypdfium2", "PIL"])
    if ocr_modules_ready and tesseract_cli_available():
        checks.append({"name": "OCR readiness", "status": "ok", "message": "OCR support is ready for scanned PDFs."})
    else:
        checks.append({"name": "OCR readiness", "status": "info", "message": "Scanned PDF OCR needs pytesseract, pypdfium2, Pillow, and the Tesseract CLI."})
    return checks
