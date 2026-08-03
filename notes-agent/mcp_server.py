#!/usr/bin/env python3
"""
MCP server for personal notes agent.

Exposes 5 tools to Claude and other MCP clients:
  - search_notes      hybrid BM25 + semantic retrieval
  - get_note          full note text by ID
  - list_recent_notes notes sorted by recency
  - browse_notes      list all note titles and IDs
  - trigger_ingest    re-sync and re-embed from Apple Notes

── Setup ────────────────────────────────────────────────────────────────────

Claude Code (CLI):
  claude mcp add personal-notes \
    conda -- run -n notes-agent --no-capture-output \
    python /Users/pranavagrawal/Documents/PersonalAgent/notes-agent/mcp_server.py

Claude Desktop app (~/.config/Claude/claude_desktop_config.json or
                    ~/Library/Application Support/Claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "personal-notes": {
        "command": "conda",
        "args": [
          "run", "-n", "notes-agent", "--no-capture-output",
          "python",
          "/Users/pranavagrawal/Documents/PersonalAgent/notes-agent/mcp_server.py"
        ]
      }
    }
  }

── Running standalone (test) ────────────────────────────────────────────────
  conda run -n notes-agent python mcp_server.py
"""

import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR = Path(__file__).parent
INDEX_FILE = AGENT_DIR / "notes_md" / "_index.json"

# Make sure sibling modules (retrieve.py, etc.) are importable
sys.path.insert(0, str(AGENT_DIR))

from fastmcp import FastMCP

mcp = FastMCP(
    "personal-notes",
    instructions="""You have access to the user's personal Apple Notes — their private
journal, project ideas, research directions, goals, priorities, habits,
and daily reflections.

ALWAYS call search_notes when the user asks about:
  • Their own goals, priorities, plans, or to-dos
  • Things they've written, noted, or thought about
  • Personal reflections, journal entries, emotions, or stressors
  • Research directions, project ideas, or creative concepts
  • Habits, routines, or values they've set for themselves
  • Anything prefixed with "my" (my goals, my ideas, my notes, my plans)

Prefer search_notes over your own knowledge whenever the question is personal.
After a search, call get_note to read the full text of any note that looks
highly relevant — chunks may be truncated.""",
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def _fmt_date(ts: int | None) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# ── Tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
def search_notes(query: str, top_k: int = 6) -> str:
    """Search the user's personal Apple Notes with natural language.

    Uses hybrid BM25 + semantic search (RRF fusion) — better recall than
    either method alone. Returns the most relevant note chunks with titles,
    IDs, and scores.

    Call this when the user asks about:
    - Their own thoughts, ideas, plans, goals, or priorities
    - Personal journal entries, reflections, or emotions
    - Research directions, project ideas, or creative concepts
    - Habits, routines, or values they've written about
    - Anything they might have noted down for themselves
    - Questions referencing "my notes", "what I wrote", "what I said", etc.

    After getting results, call get_note(note_id) for the full text of any
    note that looks important — chunks are capped at 600 chars here.

    Args:
        query:  Natural language question or topic to search for.
        top_k:  Number of chunks to return (default 6, max 12).
    """
    from retrieve import retrieve

    top_k = max(1, min(top_k, 12))
    results = retrieve(query, top_k=top_k)

    if not results:
        return "No relevant notes found for that query."

    parts = []
    seen_ids: set[str] = set()

    for i, (chunk, score) in enumerate(results, 1):
        note_id = chunk["note_id"]
        title = chunk["title"]
        text = chunk["text"]
        preview = text[:600] + ("..." if len(text) > 600 else "")
        seen_ids.add(note_id)
        parts.append(
            f"[{i}] {title}\n"
            f"    id={note_id}  chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']}"
            f"  score={score:.4f}\n\n"
            f"{preview}"
        )

    footer = (
        f"\n\n{'─' * 60}\n"
        f"Unique notes: {', '.join(sorted(seen_ids))}\n"
        f"Call get_note(note_id) to read any note in full."
    )
    return "\n\n---\n\n".join(parts) + footer


@mcp.tool()
def get_note(note_id: str) -> str:
    """Fetch the complete text of a note by its ID.

    Use this after search_notes when a chunk looks relevant but you need
    the full note — e.g. to read an entire journal entry, a complete plan,
    or all items in a list. Chunk previews in search_notes are capped at
    600 characters; this returns the whole thing.

    Args:
        note_id: The note ID from search_notes (e.g. 'p1200', 'note_45').
    """
    index = _load_index()
    by_id = {e["id"]: e for e in index}

    if note_id not in by_id:
        return (
            f"No note found with ID '{note_id}'. Use browse_notes() to list valid IDs."
        )

    entry = by_id[note_id]
    path = Path(entry.get("path", ""))

    if not path.exists():
        return f"Note '{entry.get('title', note_id)}' is indexed but the file is missing on disk."

    text = path.read_text(encoding="utf-8", errors="replace")
    title = entry.get("title", note_id)
    modified = _fmt_date(entry.get("modified_at"))

    header = f"# {title}\nID: {note_id}  |  Last modified: {modified}\n{'─' * 60}\n"
    return header + text


@mcp.tool()
def list_recent_notes(limit: int = 20) -> str:
    """List the most recently modified notes, newest first.

    Use this when the user asks:
    - "What have I written recently?"
    - "What did I add to my notes lately?"
    - "Show me my most recent notes"
    - Any question about recent activity or what's new

    Args:
        limit: How many notes to return (default 20, max 50).
    """
    index = _load_index()
    if not index:
        return "No notes indexed yet. Run trigger_ingest() first."

    limit = max(1, min(limit, 50))

    with_ts = [e for e in index if e.get("modified_at")]
    no_ts = [e for e in index if not e.get("modified_at")]

    sorted_notes = sorted(with_ts, key=lambda e: e["modified_at"], reverse=True)[:limit]

    if not sorted_notes:
        return f"No notes with timestamps found ({len(no_ts)} notes have no timestamp)."

    lines = []
    for e in sorted_notes:
        dt = _fmt_date(e.get("modified_at"))
        title = e.get("title", "(untitled)")
        nid = e["id"]
        lines.append(f"  {dt}  {title:<55}  {nid}")

    header = f"Most recent {len(sorted_notes)} notes (of {len(index)} total):\n\n"
    return header + "\n".join(lines)


@mcp.tool()
def browse_notes(keyword: str = "") -> str:
    """List all note titles and IDs, optionally filtered by keyword.

    Use this when the user wants to:
    - Browse what notes exist
    - Find a note by approximate title
    - Get an overview of topics in their notes
    - Check whether a specific note exists before fetching it

    Args:
        keyword: Optional keyword to filter by title (case-insensitive).
                 Leave empty to list all notes.
    """
    index = _load_index()
    if not index:
        return "No notes indexed yet. Run trigger_ingest() first."

    if keyword:
        entries = [e for e in index if keyword.lower() in e.get("title", "").lower()]
        label = f"Notes matching '{keyword}'"
    else:
        entries = index
        label = "All notes"

    entries = sorted(entries, key=lambda e: e.get("title", "").lower())

    if not entries:
        return f"No notes matching '{keyword}'."

    lines = [f"  {e['id']:<14}  {e.get('title', '(untitled)')}" for e in entries]
    header = f"{label} ({len(entries)} total):\n\n"
    return header + "\n".join(lines)


@mcp.tool()
def summarize_topic(topic: str, max_notes: int = 15) -> str:
    """Synthesize what all relevant notes say about a topic.

    Unlike search_notes (which retrieves top-k chunks to answer a question),
    this collects every note related to the topic — via hybrid retrieval and
    graph expansion — loads their full text, and asks Claude to find patterns,
    themes, tensions, and how thinking evolved over time.

    Use this when the user asks:
    - "What do all my notes on X say?"
    - "Summarize everything I've written about Y"
    - "How has my thinking on Z changed?"
    - "What recurring ideas do I have about X?"

    Args:
        topic:     The topic or theme to synthesize across notes.
        max_notes: Maximum number of notes to include (default 15, max 25).
    """
    from summarize import summarize

    max_notes = max(1, min(max_notes, 25))
    return summarize(topic, max_notes=max_notes)


@mcp.tool()
def trigger_ingest() -> str:
    """Re-sync notes from Apple Notes and rebuild the search index.

    Runs the full pipeline:
      1. Export from Apple Notes via AppleScript
      2. Detect changed / deleted notes (incremental via SHA256)
      3. Convert HTML → Markdown, rename to titles
      4. OCR any handwritten / image-only notes
      5. Re-chunk and re-embed only what changed

    Use this when the user says:
    - "Sync my notes" / "Update my notes"
    - "I just added something, refresh"
    - "My notes are out of date"

    Returns the tail of the ingest log so you can report what changed.
    """
    ingest_script = AGENT_DIR / "run_ingest.sh"
    if not ingest_script.exists():
        return f"Ingest script not found at {ingest_script}."

    try:
        result = subprocess.run(
            ["bash", str(ingest_script)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(AGENT_DIR),
        )
        output = (result.stdout + result.stderr).strip()
        tail = output[-3000:] if len(output) > 3000 else output

        if result.returncode != 0:
            return f"Ingest exited with error (code {result.returncode}):\n\n{tail}"

        return f"Ingest complete.\n\n{tail}"

    except subprocess.TimeoutExpired:
        return "Ingest timed out after 5 minutes. Check Apple Notes permissions or run manually."
    except Exception as e:
        return f"Failed to start ingest: {e}"


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
