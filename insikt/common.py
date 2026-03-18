from __future__ import annotations

import html
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    replacements = {
        "??": "?",
        "??": "?",
        "??": "?",
        "??": "?",
        "??": "?",
        "??": "?",
        "â€“": "-",
        "â€”": "-",
        "â€˜": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¦": "...",
        "M??STE": "M?STE",
        "k??lla": "k?lla",
        "K??lla": "K?lla",
        "Ber??knar": "Ber?knar",
        "T??nker": "T?nker",
        "??vers??tt": "?vers?tt",
        "??vers??tter": "?vers?tter",
        "??vers??ttning": "?vers?ttning",
        "f??r": "f?r",
        "Fr??n": "Fr?n",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def strip_emoji(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "So")


def cleaned_ui_text(text: str) -> str:
    return strip_emoji(normalize_text(text))


def safe_html_fragment(text: str) -> str:
    escaped = html.escape(text or "")
    return escaped.replace("\n", "<br>")


def docs_to_records(docs: Sequence[Document]) -> List[dict]:
    return [{"page_content": doc.page_content, "metadata": dict(doc.metadata)} for doc in docs]


def records_to_docs(records: Sequence[dict]) -> List[Document]:
    return [Document(page_content=record.get("page_content", ""), metadata=record.get("metadata", {})) for record in records]


def chat_to_records(chat_history: Sequence[dict]) -> List[dict]:
    return [{"role": str(item.get("role", "assistant")), "content": str(item.get("content", ""))} for item in chat_history]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._ -]+", "", value).strip()
    value = re.sub(r"\s+", "-", value)
    return value[:64] or "save-slot"


def compute_text_hash(parts: Iterable[str]) -> str:
    hasher = hashlib.md5()
    for part in parts:
        hasher.update((part or "").encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def compute_uploaded_files_fingerprint(uploaded_files: Sequence) -> str:
    hasher = hashlib.md5()
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        hasher.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        hasher.update(str(len(content)).encode("ascii"))
        hasher.update(content)
    return hasher.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
