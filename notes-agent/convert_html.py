#!/usr/bin/env python3
"""
Step 1: Convert HTML notes to Markdown + build _index.json manifest.

Reads:  notes_export/manifest.txt  (note_N|||title)
        notes_export/note_N.html

Writes: notes_md/note_N.md
        notes_md/_index.json        (title, path, char_count per note)
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    import html2text
except ImportError:
    print("Missing deps. Run: pip3 install beautifulsoup4 html2text")
    sys.exit(1)

from utils import load_manifest

AGENT_DIR  = Path(__file__).parent
EXPORT_DIR = Path("~/Desktop/notes_export").expanduser()
OUT_DIR    = AGENT_DIR / "notes_md"

# Fallback to sibling notes_export if Desktop one doesn't exist (dev convenience)
if not EXPORT_DIR.exists():
    EXPORT_DIR = AGENT_DIR.parent / "notes_export"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- html2text config ---
h = html2text.HTML2Text()
h.ignore_links   = False
h.ignore_images  = True
h.body_width     = 0        # don't hard-wrap lines
h.ul_item_mark   = "-"


def html_to_md(html: str) -> str:
    # bs4 first pass: normalise Apple Notes quirks
    soup = BeautifulSoup(html, "html.parser")

    # Apple Notes wraps everything in <div>; collapse consecutive <br> / empty divs
    for tag in soup.find_all("div"):
        if tag.get_text(strip=True) == "":
            tag.decompose()

    cleaned = str(soup)
    md = h.handle(cleaned)

    # Collapse 3+ blank lines → 2
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()



def sanity_check(note_id: str, title: str, md: str, html: str) -> list[str]:
    warnings = []
    if len(md) == 0:
        if "data:image/" in html:
            warnings.append("IMAGE_ONLY")
        else:
            warnings.append("EMPTY")
    if len(md) > 100_000:
        warnings.append(f"OVERSIZED ({len(md):,} chars)")
    # Garbled: high ratio of non-ASCII could indicate encoding issues
    non_ascii = sum(1 for c in md if ord(c) > 127)
    if len(md) > 50 and non_ascii / len(md) > 0.3:
        warnings.append(f"HIGH_NON_ASCII ({non_ascii/len(md):.0%})")
    return warnings


def main():
    manifest_path = EXPORT_DIR / "manifest.txt"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    index = []
    issues = []

    for note_id, meta in manifest.items():
        title       = meta["title"]
        modified_at = meta["modified_at"]

        html_path = EXPORT_DIR / f"{note_id}.html"
        if not html_path.exists():
            print(f"  WARN: {note_id}.html missing, skipping")
            continue

        html = html_path.read_text(encoding="utf-8", errors="replace")
        md   = html_to_md(html)

        warns = sanity_check(note_id, title, md, html)
        if warns:
            issues.append((note_id, title, warns))

        # Always use manifest title as H1, stripping any H1 from html2text
        if md.startswith("#"):
            parts = md.split("\n", 1)
            md = parts[1].lstrip("\n") if len(parts) > 1 else ""
        md = f"# {title}\n\n{md}" if md else f"# {title}"

        out_path = OUT_DIR / f"{note_id}.md"
        out_path.write_text(md, encoding="utf-8")

        index.append({
            "id":          note_id,
            "title":       title,
            "modified_at": modified_at,
            "path":        str(out_path),
            "char_count":  len(md),
        })

    # Write index atomically to avoid partial writes on crash
    index_path = OUT_DIR / "_index.json"
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, index_path)

    print(f"Converted {len(index)} notes → {OUT_DIR}")
    print(f"Index written → {index_path}")

    if issues:
        print(f"\nSanity check warnings ({len(issues)} notes):")
        for note_id, title, warns in issues:
            print(f"  {note_id} [{title[:40]}]: {', '.join(warns)}")


if __name__ == "__main__":
    main()
