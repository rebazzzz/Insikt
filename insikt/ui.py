from __future__ import annotations


def app_readiness_label(docs, processing: bool) -> tuple[str, str]:
    if processing:
        return "processing", "Processing in progress"
    if docs:
        return "ready", "Knowledge base ready"
    return "idle", "No knowledge base loaded"


def concise_model_label(model_info: dict, lang: str) -> str:
    name = model_info.get("display_name_en" if lang == "en" else "display_name", "")
    speed = model_info.get("speed", "")
    quality = model_info.get("quality", "")
    return " | ".join(part for part in [name, speed, quality] if part)
