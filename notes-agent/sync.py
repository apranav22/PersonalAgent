#!/usr/bin/env python3
"""
Step 2: Incremental sync via SHA256 hashing.

Compares current HTML files against stored hashes (_hashes.json) to produce:
  - changed_ids:  notes added or modified  → need re-embed
  - deleted_ids:  notes removed from export → need removal from vector store

Writes updated _hashes.json after each run.

Usage (standalone):
    python3 sync.py             # prints diff, updates hashes
    python3 sync.py --dry-run   # prints diff only, no writes

In the pipeline, import get_diff() and pass results downstream.
"""

import argparse
import hashlib
import json
import sys

from pathlib import Path

from utils import load_manifest

AGENT_DIR   = Path(__file__).parent
EXPORT_DIR  = Path("~/Desktop/notes_export").expanduser()
HASHES_FILE = AGENT_DIR / "_hashes.json"

# Fallback to sibling notes_export if Desktop one doesn't exist (dev convenience)
if not EXPORT_DIR.exists():
    EXPORT_DIR = AGENT_DIR.parent / "notes_export"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hashes() -> dict[str, str]:
    if HASHES_FILE.exists():
        return json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    return {}


def save_hashes(hashes: dict[str, str]) -> None:
    HASHES_FILE.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def get_diff(dry_run: bool = False) -> tuple[list[str], list[str]]:
    """
    Returns (changed_ids, deleted_ids).
    changed_ids: note IDs that are new or whose content changed.
    deleted_ids: note IDs present in previous hashes but no longer on disk.
    Updates _hashes.json unless dry_run=True.
    """
    old_hashes = load_hashes()
    new_hashes: dict[str, str] = {}

    manifest_path = EXPORT_DIR / "manifest.txt"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    # Collect current note IDs from manifest
    manifest = load_manifest(manifest_path)
    current_ids = set(manifest.keys())

    changed_ids: list[str] = []
    for note_id in current_ids:
        html_path = EXPORT_DIR / f"{note_id}.html"
        if not html_path.exists():
            continue
        h = sha256(html_path)
        new_hashes[note_id] = h
        if old_hashes.get(note_id) != h:
            changed_ids.append(note_id)

    deleted_ids = [nid for nid in old_hashes if nid not in current_ids]

    if not dry_run:
        save_hashes(new_hashes)

    return changed_ids, deleted_ids


def main():
    parser = argparse.ArgumentParser(description="Incremental notes sync")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing hashes")
    parser.add_argument("--json",    action="store_true", help="Output as JSON for pipeline use")
    args = parser.parse_args()

    changed, deleted = get_diff(dry_run=args.dry_run)

    if args.json:
        print(json.dumps({"changed": changed, "deleted": deleted}))
        return

    print(f"Changed/new: {len(changed)}")
    for nid in changed:
        print(f"  + {nid}")

    print(f"Deleted:     {len(deleted)}")
    for nid in deleted:
        print(f"  - {nid}")

    if args.dry_run:
        print("\n[dry-run] _hashes.json not updated")
    else:
        print(f"\n_hashes.json updated → {HASHES_FILE}")


if __name__ == "__main__":
    main()
