from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from .common import chat_to_records, docs_to_records, read_json, records_to_docs, slugify, write_json


def list_save_slots(root: Path) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    slots = []
    for item in root.iterdir():
        manifest = item / "manifest.json"
        if item.is_dir() and manifest.exists():
            payload = read_json(manifest, {})
            payload["slot_id"] = item.name
            slots.append(payload)
    slots.sort(key=lambda slot: slot.get("updated_at", ""), reverse=True)
    return slots


def save_slot(root: Path, slot_name: str, docs, raw_pages, chat_history, last_summary, lang, vectorstore, fingerprint: str = "") -> str:
    slot_id = slugify(slot_name)
    slot_dir = root / slot_id
    slot_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        slot_dir / "manifest.json",
        {
            "name": slot_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "lang": lang,
            "doc_count": len(docs or []),
            "raw_page_count": len(raw_pages or []),
            "chat_count": len(chat_history or []),
            "fingerprint": fingerprint,
            "has_summary": bool(last_summary),
        },
    )
    write_json(slot_dir / "docs.json", docs_to_records(docs or []))
    write_json(slot_dir / "raw_pages.json", docs_to_records(raw_pages or []))
    write_json(slot_dir / "chat_history.json", chat_to_records(chat_history or []))
    (slot_dir / "summary.txt").write_text(last_summary or "", encoding="utf-8")
    if vectorstore:
        vector_dir = slot_dir / "vectorstore"
        vector_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vector_dir))
    return slot_id


def load_slot(root: Path, slot_id: str, embeddings):
    from langchain_community.vectorstores import FAISS

    slot_dir = root / slot_id
    manifest = read_json(slot_dir / "manifest.json", {})
    docs = records_to_docs(read_json(slot_dir / "docs.json", []))
    raw_pages = records_to_docs(read_json(slot_dir / "raw_pages.json", []))
    chat_history = read_json(slot_dir / "chat_history.json", [])
    summary = (slot_dir / "summary.txt").read_text(encoding="utf-8") if (slot_dir / "summary.txt").exists() else ""
    vectorstore = None
    vector_dir = slot_dir / "vectorstore"
    if (vector_dir / "index.faiss").exists() and (vector_dir / "index.pkl").exists():
        vectorstore = FAISS.load_local(str(vector_dir), embeddings, allow_dangerous_deserialization=True)
    return {
        "manifest": manifest,
        "docs": docs,
        "raw_pages": raw_pages,
        "chat_history": chat_history,
        "last_summary": summary,
        "vectorstore": vectorstore,
        "fingerprint": manifest.get("fingerprint", ""),
        "lang": manifest.get("lang", "sv"),
    }


def delete_slot(root: Path, slot_id: str) -> None:
    slot_dir = root / slot_id
    if slot_dir.exists():
        shutil.rmtree(slot_dir)
