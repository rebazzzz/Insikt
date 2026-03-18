"""Insikt application modules."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

__version__ = "0.3.0-early"
BUILD_CHANNEL = "early-tester"


def get_build_metadata() -> dict[str, str]:
    commit = "local"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
    except Exception:
        commit = "local"
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "version": __version__,
        "channel": BUILD_CHANNEL,
        "commit": commit,
        "built_at": built_at,
        "label": f"{__version__} | {BUILD_CHANNEL} | {commit} | {built_at}",
    }
