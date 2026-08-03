#!/usr/bin/env python3
"""
Rename notes_md/note_N.md → notes_md/{title}.md using manifest titles.
Also updates _index.json paths.

Collision handling: if two notes share a sanitized title, appends _noteN.
"""

import json
import re
import sys
from pathlib import Path

from utils import load_manifest as _load_manifest

AGENT_DIR  = Path(__file__).parent
NOTES_DIR  = AGENT_DIR / "notes_md"
INDEX_FILE = NOTES_DIR / "_index.json"

EXPORT_DIR = Path("~/Desktop/notes_export").expanduser()
if not EXPORT_DIR.exists():
    EXPORT_DIR = AGENT_DIR.parent / "notes_export"

MANIFEST = EXPORT_DIR / "manifest.txt"


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)       # remove punctuation except - and _
    s = re.sub(r"[\s_]+", "_", s)         # spaces → underscore
    s = re.sub(r"-+", "-", s)
    s = s.strip("_-")
    return s[:80] or "untitled"           # cap length


def load_manifest() -> dict[str, str]:
    """Returns {note_id: title} for use in renaming."""
    raw = _load_manifest(MANIFEST)
    return {note_id: meta["title"] for note_id, meta in raw.items()}


def main():
    if not INDEX_FILE.exists():
        print("_index.json not found — run convert_html.py first")
        sys.exit(1)

    manifest = load_manifest()
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    seen_slugs: dict[str, str] = {}  # slug → note_id that claimed it first
    renames = 0

    for entry in index:
        note_id = entry["id"]
        title   = manifest.get(note_id, note_id)
        slug    = slugify(title)

        # Collision: append note_id to disambiguate
        if slug in seen_slugs:
            slug = f"{slug}_{note_id}"
        seen_slugs[slug] = note_id

        old_path = NOTES_DIR / f"{note_id}.md"
        new_path = NOTES_DIR / f"{slug}.md"

        if not old_path.exists():
            # Already renamed in a previous run — find current file
            current = Path(entry["path"])
            if current.exists() and current != new_path:
                try:
                    current.rename(new_path)
                    entry["path"] = str(new_path)
                    renames += 1
                except OSError as e:
                    print(f"  WARN: could not rename {current} → {new_path}: {e}")
            elif current.exists():
                entry["path"] = str(new_path)
            continue

        if old_path != new_path:
            try:
                old_path.rename(new_path)
                renames += 1
            except OSError as e:
                print(f"  WARN: could not rename {old_path} → {new_path}: {e}")

        entry["path"] = str(new_path)

    INDEX_FILE.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Renamed {renames} files → {NOTES_DIR}")
    print(f"_index.json updated")


if __name__ == "__main__":
    main()
