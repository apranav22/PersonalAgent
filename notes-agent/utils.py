#!/usr/bin/env python3
"""
Shared utilities for the notes-agent pipeline.
"""

from pathlib import Path


def load_manifest(path: Path) -> dict:
    """
    Parse a manifest.txt file and return note metadata.

    Returns {note_id: {"title": str, "modified_at": int|None}}

    Format: note_id|||title[|||modified_at_float]
    """
    mapping = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|||")
        if len(parts) >= 2:
            note_id = parts[0].strip()
            title   = parts[1].strip()
            modified_at = None
            if len(parts) >= 3:
                try:
                    modified_at = int(float(parts[2].strip()))
                except Exception:
                    modified_at = None
            mapping[note_id] = {"title": title, "modified_at": modified_at}
    return mapping
