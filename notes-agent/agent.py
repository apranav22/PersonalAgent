#!/usr/bin/env python3
"""
Personal notes agent — query your Apple Notes with natural language.

Retrieves relevant chunks via hybrid search (BM25 + semantic),
sends them to Claude as context, returns a grounded answer with citations.

Usage:
    python3 agent.py "what did I write about confidence?"
    python3 agent.py "what are my project ideas?" --top-k 8
    python3 agent.py --chat     # interactive mode
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

AGENT_DIR = Path(__file__).parent

SYSTEM_PROMPT = """You are a personal notes assistant. You have access to the user's Apple Notes, retrieved as context below.

Your job:
- Answer the user's question using ONLY information from the provided notes
- Be specific and concrete — quote or paraphrase directly from the notes where useful
- Cite your sources: after each claim, add the note title in brackets e.g. [confidence journal]
- If multiple notes say related things, synthesize them
- If the notes don't contain enough to answer the question, say so clearly — don't hallucinate

The notes are personal and informal. Treat them as-is."""


def build_context(results: list) -> str:
    parts = []
    for i, (chunk, score) in enumerate(results, 1):
        parts.append(
            f"--- Note {i}: {chunk['title']} ---\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def get_api_key() -> str:
    import subprocess
    # 1. Try macOS Keychain
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", os.environ["USER"], "-s", "ANTHROPIC_API_KEY", "-w"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # 2. Fall back to environment variable
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    print("No API key found. Store it with:")
    print('  security add-generic-password -a "$USER" -s "ANTHROPIC_API_KEY" -w "sk-ant-..."')
    sys.exit(1)


def ask_claude(context: str, query: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key())
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<notes>\n{context}\n</notes>\n\n{query}"}]
    )
    return message.content[0].text


def ask_ollama(context: str, query: str, model: str = "qwen2.5:7b") -> str:
    import urllib.request, json
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"<notes>\n{context}\n</notes>\n\n{query}"}
        ]
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]


def ask(query: str, top_k: int = 6, local: bool = False) -> str:
    from retrieve import retrieve

    results = retrieve(query, top_k=top_k)
    if not results:
        return "No relevant notes found."

    context = build_context(results)
    answer  = ask_ollama(context, query) if local else ask_claude(context, query)

    seen = set()
    sources = []
    for chunk, _ in results:
        if chunk["note_id"] not in seen:
            seen.add(chunk["note_id"])
            sources.append(chunk["title"])
    source_lines = "\n".join(f"  • {title}" for title in sources)

    return f"{answer}\n\n**Sources:**\n{source_lines}"


def chat_loop(top_k: int, local: bool = False):
    provider = "qwen2.5:7b (local)" if local else "claude-haiku (API)"
    print(f"Personal Notes Agent [{provider}] — type 'quit' to exit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        print("\nAgent:", ask(query, top_k=top_k, local=local), "\n")


def lookup(note_id: str) -> None:
    import json
    index = json.loads((AGENT_DIR / "notes_md/_index.json").read_text())
    by_id = {e["id"]: e for e in index}
    if note_id not in by_id:
        print(f"No note found with ID: {note_id}")
        return
    e = by_id[note_id]
    path = Path(e.get("path", ""))
    if not path.exists():
        print(f"Note file not found on disk: {path}")
        return
    print(f"ID:    {e['id']}")
    print(f"Title: {e.get('title', '(unknown)')}")
    print(f"File:  {path}")
    print()
    print(path.read_text(encoding="utf-8", errors="replace"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query",             nargs="?", help="Your question")
    parser.add_argument("--top-k",  type=int, default=6,    help="Chunks to retrieve")
    parser.add_argument("--chat",   action="store_true",     help="Interactive chat mode")
    parser.add_argument("--api",    action="store_true",     help="Use Claude Haiku via Anthropic API (default: local qwen2.5:7b)")
    parser.add_argument("--lookup",                          help="Show note by ID e.g. --lookup p1200")
    args = parser.parse_args()
    local = not args.api

    if args.lookup:
        lookup(args.lookup)
    elif args.chat:
        chat_loop(args.top_k, local=local)
    elif args.query:
        print(ask(args.query, top_k=args.top_k, local=local))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
