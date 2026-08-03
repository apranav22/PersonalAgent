#!/usr/bin/env python3
"""
OCR handwritten/image-only Apple Notes using macOS Vision framework.

For each IMAGE_ONLY note:
  1. Extract base64 PNG(s) from the HTML
  2. Write to a temp file
  3. Call ./ocr_tool (compiled Swift binary) to get text
  4. Append OCR text into the note's .md file

Run after convert_html.py. Safe to re-run — skips notes already OCR'd
(detected by presence of <!-- ocr --> marker in the .md).

Usage:
    python3 ocr_notes.py              # process all image-only notes
    python3 ocr_notes.py --ids note_45 note_46   # specific notes only
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

AGENT_DIR  = Path(__file__).parent
NOTES_DIR  = AGENT_DIR / "notes_md"
INDEX_FILE = NOTES_DIR / "_index.json"
OCR_TOOL   = AGENT_DIR / "ocr_tool"

EXPORT_DIR = Path("~/Desktop/notes_export").expanduser()
if not EXPORT_DIR.exists():
    EXPORT_DIR = AGENT_DIR.parent / "notes_export"


def extract_images(html: str) -> list[bytes]:
    """Pull all base64-encoded images out of an HTML string."""
    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    images = []
    for match in re.finditer(pattern, html):
        try:
            images.append(base64.b64decode(match.group(1)))
        except Exception as e:
            print(f"  WARN: failed to decode image: {e}")
    return images


def ocr_image(image_bytes: bytes) -> str:
    """Run ocr_tool on raw image bytes, return recognized text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [str(OCR_TOOL), tmp_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"ocr_tool exited {result.returncode}: {result.stderr}")
        return result.stdout.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def is_image_only(html: str) -> bool:
    return "data:image/" in html


def already_ocrd(md_path: Path) -> bool:
    if not md_path.exists():
        return False
    return "<!-- ocr -->" in md_path.read_text(encoding="utf-8")


def process_note(note_id: str, title: str, md_path: Path) -> bool:
    html_path = EXPORT_DIR / f"{note_id}.html"
    if not html_path.exists():
        return False

    html = html_path.read_text(encoding="utf-8", errors="replace")
    if not is_image_only(html):
        return False

    if already_ocrd(md_path):
        return False

    images = extract_images(html)
    if not images:
        return False

    ocr_parts = []
    for img_bytes in images:
        text = ocr_image(img_bytes)
        if text:
            ocr_parts.append(text)

    if not ocr_parts:
        return False

    ocr_text = "\n\n".join(ocr_parts)

    # Read existing md (just the title line) and append OCR content
    existing = md_path.read_text(encoding="utf-8") if md_path.exists() else f"# {title}"
    updated = existing.rstrip() + f"\n\n<!-- ocr -->\n{ocr_text}\n"
    md_path.write_text(updated, encoding="utf-8")
    return True


def main():
    if not EXPORT_DIR.exists():
        print(f"Export directory not found: {EXPORT_DIR}")
        print("Run export_notes.scpt first.")
        sys.exit(1)

    if not OCR_TOOL.exists():
        print(f"ocr_tool binary not found at {OCR_TOOL}")
        print("Run: swiftc -O ocr_tool.swift -o ocr_tool -framework Vision -framework AppKit")
        sys.exit(1)

    if not INDEX_FILE.exists():
        print("_index.json not found — run convert_html.py first")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="Specific note IDs to OCR")
    args = parser.parse_args()

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    if args.ids:
        entries = [e for e in index if e["id"] in args.ids]
    else:
        entries = index

    done, skipped, failed = 0, 0, 0
    for entry in entries:
        note_id = entry["id"]
        title   = entry["title"]
        md_path = Path(entry["path"])

        try:
            if process_note(note_id, title, md_path):
                print(f"  OCR'd: {note_id} [{title[:40]}]")
                done += 1
            else:
                skipped += 1
        except (OSError, RuntimeError, ValueError) as e:
            print(f"  ERROR: {note_id} — {e}")
            failed += 1

    print(f"\nOCR complete: {done} processed, {skipped} skipped, {failed} errors")


if __name__ == "__main__":
    main()
