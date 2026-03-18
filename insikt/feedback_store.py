from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _slug(value: str, fallback: str = "issue") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (value or "").strip()).strip("-").lower()
    return cleaned[:48] or fallback


def _report_paths(root: Path, report_id: str) -> tuple[Path, Path]:
    created = report_id.split("-", 2)[1]
    folder = root / "reports" / created[:6]
    return folder / f"{report_id}.json", folder / f"{report_id}.md"


def build_report_id(title: str) -> str:
    stamp = _timestamp().strftime("%Y%m%d%H%M%S")
    return f"ISS-{stamp}-{_slug(title)}-{uuid.uuid4().hex[:6]}"


def render_report_markdown(payload: dict[str, Any]) -> str:
    sections = [
        f"# {payload.get('title', 'Untitled issue')}",
        f"- Report ID: {payload.get('report_id', '')}",
        f"- Created: {payload.get('created_at', '')}",
        f"- Reporter: {payload.get('reporter_name', '')}",
        f"- Severity: {payload.get('severity', '')}",
        f"- Area: {payload.get('area', '')}",
        f"- App build: {payload.get('app_version_label', '')}",
        "",
        "## What Happened",
        payload.get("what_happened", ""),
        "",
        "## Expected",
        payload.get("expected", ""),
        "",
        "## Steps To Reproduce",
        payload.get("steps", ""),
        "",
        "## Work Context",
        payload.get("work_context", ""),
        "",
        "## Attached Context",
        json.dumps(payload.get("app_context", {}), indent=2, ensure_ascii=False),
    ]
    return "\n".join(sections).strip() + "\n"


def save_issue_report(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    report_id = build_report_id(payload.get("title", "issue"))
    created_at = _timestamp().isoformat()
    full_payload = dict(payload)
    full_payload["report_id"] = report_id
    full_payload["created_at"] = created_at
    json_path, md_path = _report_paths(root, report_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_report_markdown(full_payload), encoding="utf-8")
    return full_payload


def list_issue_reports(root: Path, limit: int = 50) -> list[dict[str, Any]]:
    reports = []
    report_root = root / "reports"
    if not report_root.exists():
        return reports
    for path in report_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["json_path"] = str(path)
        md_path = path.with_suffix(".md")
        payload["markdown_path"] = str(md_path)
        reports.append(payload)
    reports.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return reports[:limit]


def build_feedback_bundle(root: Path, report_ids: list[str] | None = None) -> bytes:
    reports = list_issue_reports(root, limit=500)
    if report_ids:
        wanted = set(report_ids)
        reports = [report for report in reports if report.get("report_id") in wanted]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        manifest = []
        for report in reports:
            report_id = report.get("report_id", "unknown")
            json_path = Path(report.get("json_path", ""))
            md_path = Path(report.get("markdown_path", ""))
            if json_path.exists():
                bundle.writestr(f"reports/{json_path.name}", json_path.read_text(encoding="utf-8"))
            if md_path.exists():
                bundle.writestr(f"reports/{md_path.name}", md_path.read_text(encoding="utf-8"))
            manifest.append(
                {
                    "report_id": report_id,
                    "title": report.get("title", ""),
                    "created_at": report.get("created_at", ""),
                    "reporter_name": report.get("reporter_name", ""),
                    "severity": report.get("severity", ""),
                    "area": report.get("area", ""),
                }
            )
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return buffer.getvalue()

