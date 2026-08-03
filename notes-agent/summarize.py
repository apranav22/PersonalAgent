#!/usr/bin/env python3
"""
Cross-note summarization: synthesize what all notes on a topic say collectively.

Unlike agent.py (which answers a specific question from top-k chunks), this:
  1. Collects all notes relevant to a topic via hybrid retrieval + graph expansion
  2. Loads full note texts (not just chunks) for richer context
  3. Asks Claude to synthesize across them — finding themes, tensions, evolution

Usage (standalone):
    python3 summarize.py "confidence and self belief"
    python3 summarize.py "project ideas" --max-notes 20
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR  = Path(__file__).parent
INDEX_FILE = AGENT_DIR / "notes_md" / "_index.json"

MAX_CONTEXT_CHARS = 80_000   # ~20k tokens — well within Haiku's 200k context

SYSTEM_PROMPT = """You are synthesizing someone's personal notes on a topic.

Do NOT summarize each note one by one. Instead, read across all of them and produce:

**Core themes** — ideas, feelings, or goals that keep recurring
**Tensions or shifts** — where notes contradict each other or show changing views over time
**What stands out** — the single most striking or important insight across all notes
**Sources** — bullet list of note titles used

Be direct and specific. Quote or paraphrase the notes where it adds weight.
If dates are available, use them to show how thinking evolved."""


# ── Note collection ───────────────────────────────────────────────────────


def _load_index() -> dict[str, dict]:
    if not INDEX_FILE.exists():
        return {}
    return {e["id"]: e for e in json.loads(INDEX_FILE.read_text(encoding="utf-8"))}


def _read_note(entry: dict) -> str | None:
    path = Path(entry.get("path", ""))
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def collect_notes(topic: str, max_notes: int = 15) -> list[dict]:
    """
    Gather notes relevant to topic via hybrid retrieval + 1-hop graph expansion.

    Returns list of {note_id, title, modified_at, text, score}, sorted by score.
    Full note texts are loaded (not chunks) to give Claude complete context.
    """
    from retrieve import retrieve, load_graph

    # Wider initial retrieval — we want coverage, not just top precision
    results = retrieve(topic, top_k=min(max_notes, 12), expand_graph=False)

    seen: set[str] = set()
    scored: dict[str, float] = {}

    for chunk, score in results:
        nid = chunk["note_id"]
        if nid not in seen:
            seen.add(nid)
            scored[nid] = score

    # Graph expansion — no discount cap here, we want related notes even if weaker
    graph = load_graph()
    neighbor_scores: dict[str, float] = {}
    for nid, direct_score in scored.items():
        for edge in graph.get(nid, []):
            nnid = edge["note_id"]
            if nnid in seen:
                continue
            candidate = direct_score * edge["weight"]
            neighbor_scores[nnid] = max(neighbor_scores.get(nnid, 0.0), candidate)

    # Add top neighbors up to max_notes
    for nnid, score in sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= max_notes:
            break
        seen.add(nnid)
        scored[nnid] = score

    # Load full note texts
    index = _load_index()
    notes = []
    total_chars = 0

    for nid, score in sorted(scored.items(), key=lambda x: x[1], reverse=True):
        if nid not in index:
            continue
        entry = index[nid]
        text = _read_note(entry)
        if not text:
            continue
        # Respect context budget — truncate last note if needed
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n... [truncated]"
        notes.append({
            "note_id":     nid,
            "title":       entry.get("title", nid),
            "modified_at": entry.get("modified_at"),
            "text":        text,
            "score":       score,
        })
        total_chars += len(text)

    return notes


# ── Synthesis ─────────────────────────────────────────────────────────────


def _build_context(notes: list[dict]) -> str:
    parts = []
    for i, note in enumerate(notes, 1):
        date_str = ""
        if note.get("modified_at"):
            date_str = f"  |  last modified: {datetime.fromtimestamp(note['modified_at']).strftime('%Y-%m-%d')}"
        header = f"--- Note {i}: {note['title']}{date_str} ---"
        parts.append(f"{header}\n{note['text']}")
    return "\n\n".join(parts)


def _get_api_key() -> str:
    # Import from agent.py to avoid duplication
    import sys
    sys.path.insert(0, str(AGENT_DIR))
    from agent import get_api_key
    return get_api_key()


def summarize(topic: str, max_notes: int = 15) -> str:
    """
    Synthesize what all relevant notes say about topic.
    Returns a structured markdown summary with themes, tensions, and sources.
    """
    notes = collect_notes(topic, max_notes=max_notes)

    if not notes:
        return "No relevant notes found for that topic."

    context = _build_context(notes)

    import anthropic
    client = anthropic.Anthropic(api_key=_get_api_key())
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"<notes>\n{context}\n</notes>\n\n"
                f"Synthesize what these notes say about: **{topic}**"
            ),
        }],
    )

    summary = message.content[0].text
    return f"{summary}\n\n---\n*Synthesized from {len(notes)} notes*"


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic",                       help="Topic to synthesize")
    parser.add_argument("--max-notes", type=int, default=15, help="Max notes to include")
    args = parser.parse_args()

    print(f"\nSynthesizing: \"{args.topic}\"\n{'─'*60}\n")
    print(summarize(args.topic, max_notes=args.max_notes))


if __name__ == "__main__":
    main()
