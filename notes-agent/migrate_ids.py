#!/usr/bin/env python3
"""
One-time migration: note_N IDs → stable pNNN IDs.

Matches old note_N → new pNNN by title (from both manifests),
then updates in-place:
  - notes_md/*.md  (rename files)
  - notes_md/_index.json
  - _hashes.json
  - chunks.jsonl
"""

import json
from pathlib import Path

from utils import load_manifest as _load_manifest

AGENT_DIR = Path(__file__).parent
NOTES_DIR = AGENT_DIR / "notes_md"
INDEX_FILE = NOTES_DIR / "_index.json"
HASHES_FILE = AGENT_DIR / "_hashes.json"
CHUNKS_FILE = AGENT_DIR / "chunks.jsonl"

EXPORT_DIR = Path("~/Desktop/notes_export").expanduser()
if not EXPORT_DIR.exists():
    EXPORT_DIR = AGENT_DIR.parent / "notes_export"


def load_manifest(path: Path) -> dict:
    """Returns {id: title}"""
    raw = _load_manifest(path)
    return {note_id: meta["title"] for note_id, meta in raw.items()}


def main():
    # Load old manifest from _index.json (source of truth for old IDs)
    if not INDEX_FILE.exists():
        print("_index.json not found")
        return

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    old_by_title = {e["title"]: e["id"] for e in index}

    # Load new manifest
    new_manifest = load_manifest(EXPORT_DIR / "manifest.txt")
    new_by_title = {title: nid for nid, title in new_manifest.items()}

    # Build prefix index for fuzzy matching on truncated titles
    # Mark ambiguous prefixes (shared by multiple titles) as None
    new_by_prefix = {}
    for t, nid in new_by_title.items():
        prefix = t[:50]
        new_by_prefix[prefix] = None if prefix in new_by_prefix else nid

    # Build mapping: old_id → new_id
    id_map = {}
    unmatched = []
    for title, old_id in old_by_title.items():
        if title in new_by_title:
            id_map[old_id] = new_by_title[title]
        elif title[:50] in new_by_prefix and new_by_prefix[title[:50]] is not None:
            id_map[old_id] = new_by_prefix[title[:50]]
        else:
            unmatched.append((old_id, title))

    print(f"Matched {len(id_map)} notes, {len(unmatched)} unmatched")
    if unmatched:
        for old_id, title in unmatched[:10]:
            print(f"  UNMATCHED: {old_id} [{title}]")

    # 1. Rename .md files
    renamed = 0
    for entry in index:
        old_id = entry["id"]
        new_id = id_map.get(old_id)
        if not new_id:
            continue
        old_path = Path(entry["path"])
        if not old_path.exists():
            continue
        # Keep the slug filename, just update the content's note_id reference
        # (filenames are title-based so they don't need renaming)
        renamed += 1

    print(f"MD files: {renamed} matched (filenames unchanged — title-based)")

    # 2. Update _index.json
    for entry in index:
        old_id = entry["id"]
        new_id = id_map.get(old_id)
        if new_id:
            entry["id"] = new_id
    INDEX_FILE.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Updated _index.json")

    # 3. Update _hashes.json
    if HASHES_FILE.exists():
        old_hashes = json.loads(HASHES_FILE.read_text(encoding="utf-8"))
        new_hashes = {}
        for old_id, hash_val in old_hashes.items():
            new_id = id_map.get(old_id, old_id)
            new_hashes[new_id] = hash_val
        HASHES_FILE.write_text(json.dumps(new_hashes, indent=2), encoding="utf-8")
        print("Updated _hashes.json")

    # 4. Update chunks.jsonl
    if CHUNKS_FILE.exists():
        lines = []
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            old_id = chunk["note_id"]
            chunk["note_id"] = id_map.get(old_id, old_id)
            lines.append(json.dumps(chunk, ensure_ascii=False))
        CHUNKS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Updated chunks.jsonl")

    print("\nMigration complete. Run embed.py --all to rebuild vectors with new IDs.")


if __name__ == "__main__":
    main()
