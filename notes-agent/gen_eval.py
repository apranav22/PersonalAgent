#!/usr/bin/env python3
"""
Auto-generate eval queries from actual notes using Claude.

Samples notes from the corpus, asks Claude to generate realistic natural-language
queries a user would ask, and identifies which note IDs are relevant.

Output: eval_queries_generated.json  (same format as eval_queries.json)

Usage:
    python3 gen_eval.py --count 40
    python3 gen_eval.py --count 40 --merge   # merge into eval_queries.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

AGENT_DIR  = Path(__file__).parent
INDEX_FILE = AGENT_DIR / "notes_md" / "_index.json"
OUT_FILE   = AGENT_DIR / "eval_queries_generated.json"
EVAL_FILE  = AGENT_DIR / "eval_queries.json"

# Only sample notes with enough content to generate good queries
MIN_CHARS = 80


def load_notes(min_chars: int = MIN_CHARS) -> list[dict]:
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    notes = []
    for e in index:
        path = Path(e.get("path", ""))
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) >= min_chars:
            notes.append({**e, "text": text})
    return notes


def generate_queries_for_notes(notes_batch: list[dict], api_key: str) -> list[dict]:
    """
    Ask Claude to generate one query per note, plus identify cross-note relevance.
    Returns list of {query, relevant_note_ids}.
    """
    import anthropic

    notes_context = ""
    for n in notes_batch:
        snippet = n["text"][:600]
        notes_context += f"\nID: {n['id']}\nTitle: {n['title']}\nContent:\n{snippet}\n---"

    prompt = f"""You are building an evaluation set for a personal notes retrieval system.

Below are {len(notes_batch)} personal notes. For each note, generate ONE natural-language query that a user might ask to find information in that note. Then, for each query, list ALL note IDs from the batch that are relevant (not just the primary one).

Rules:
- Queries should sound like real questions, not note titles paraphrased
- Queries should be answerable from the note content
- Mark a note as relevant if it contains useful information for the query, even partially
- Vary query style: some factual ("what did I write about X"), some reflective ("how do I feel about Y"), some action-oriented ("what are my plans for Z")

Output ONLY valid JSON — a list of objects:
[
  {{"query": "...", "relevant_note_ids": ["id1", "id2"]}},
  ...
]

Notes:
{notes_context}"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    return json.loads(raw.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40, help="Total queries to generate")
    parser.add_argument("--batch", type=int, default=8,  help="Notes per Claude call (default 8)")
    parser.add_argument("--merge", action="store_true",  help="Merge into eval_queries.json")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed for note sampling")
    args = parser.parse_args()

    # Get API key
    sys.path.insert(0, str(AGENT_DIR))
    from agent import get_api_key
    api_key = get_api_key()

    print(f"Loading notes...")
    notes = load_notes()
    print(f"  {len(notes)} notes with sufficient content")

    random.seed(args.seed)
    sampled = random.sample(notes, min(args.count, len(notes)))

    all_queries = []
    batches = [sampled[i:i+args.batch] for i in range(0, len(sampled), args.batch)]

    print(f"Generating queries in {len(batches)} batches of {args.batch}...")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}/{len(batches)}...", end=" ", flush=True)
        try:
            queries = generate_queries_for_notes(batch, api_key)
            all_queries.extend(queries)
            print(f"{len(queries)} queries")
        except (json.JSONDecodeError, Exception) as e:
            print(f"ERROR: {e}")
            continue

    # Deduplicate by query text
    seen = set()
    deduped = []
    for q in all_queries:
        key = q["query"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    print(f"\nGenerated {len(deduped)} unique queries")
    OUT_FILE.write_text(json.dumps(deduped, indent=2, ensure_ascii=False))
    print(f"Saved → {OUT_FILE}")

    if args.merge:
        existing = json.loads(EVAL_FILE.read_text()) if EVAL_FILE.exists() else []
        existing_qs = {q["query"].lower().strip() for q in existing}
        new_only = [q for q in deduped if q["query"].lower().strip() not in existing_qs]
        merged = existing + new_only
        EVAL_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"Merged {len(new_only)} new queries into {EVAL_FILE} ({len(merged)} total)")


if __name__ == "__main__":
    main()
