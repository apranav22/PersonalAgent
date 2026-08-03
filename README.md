# Notes Agent

A local RAG pipeline for querying Apple Notes with natural language. Fully private by default — embeddings and retrieval run on-device. Claude (API) is optional, used only for answer generation.

Built as an exploration of hybrid retrieval, graph-based note connectivity, and end-to-end RAG evaluation on real personal data.

---

## Features

- **Hybrid retrieval** — BM25 + dense vector search (all-MiniLM-L6-v2) fused via Reciprocal Rank Fusion
- **Note-level similarity graph** — cosine similarity over averaged chunk embeddings; used for retrieval expansion and cross-note synthesis
- **Cross-note summarization** — collects all notes on a topic via retrieval + graph traversal, synthesizes themes and temporal evolution
- **Handwriting OCR** — image-only notes processed via macOS Vision (Swift binary)
- **Incremental pipeline** — SHA256-based change detection; only modified notes flow through chunking, embedding, and graph rebuild
- **MCP server** — 6 tools exposed to Claude Desktop and Claude Code via FastMCP
- **Dual inference** — local Ollama (qwen2.5:7b, default) or Claude Haiku via API
- **Eval harness** — retrieval metrics (MRR, nDCG@k, P@k, R@k) + LLM-as-judge answer quality

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingest Pipeline                      │
│                                                             │
│  Apple Notes ──► AppleScript export ──► SHA256 diff         │
│       │                                      │              │
│       ▼                                      ▼              │
│  HTML ──► Markdown    (changed notes only)                  │
│       │                                                     │
│       ├──► OCR (Vision framework, image-only notes)         │
│       ├──► Chunk  (heading-aware, 50–500 tokens)            │
│       ├──► Embed  (all-MiniLM-L6-v2 → LanceDB)              │
│       └──► Graph  (pairwise cosine similarity, incremental) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Query Pipeline                        │
│                                                             │
│  Query ──► BM25 search ──┐                                  │
│        └──► Embed + ANN ──┴──► RRF fusion ──► Graph expand  │
│                                                    │        │
│                                                    ▼        │
│                              Claude Haiku / Ollama (local)  │
│                                         │                   │
│                                         ▼                   │
│                              Grounded answer + citations    │
└─────────────────────────────────────────────────────────────┘
```

---

## Evaluation

### Retrieval quality (50 queries, k=5)

| Strategy     | MRR   | nDCG@5 | Recall@10 |
|--------------|-------|--------|-----------|
| BM25 only    | 0.373 | 0.357  | 0.473     |
| Semantic only| 0.424 | 0.393  | 0.593     |
| **Hybrid**   | **0.432** | **0.402** | **0.633** |

Hybrid outperforms BM25-only by **+16% MRR** and **+34% Recall@10**.

Eval set built by sampling notes from the live corpus and generating queries with Claude, then manually verifying ground truth.

### Answer quality — LLM-as-judge (Claude Sonnet scoring Claude Haiku, 15 queries)

| Dimension      | Avg  | Median |
|----------------|------|--------|
| Faithfulness   | 4.93/5 | 5.0  |
| Relevance      | 4.87/5 | 5.0  |
| Completeness   | 4.67/5 | 5.0  |
| **Composite**  | **96.4%** |    |

Near-perfect faithfulness (4.93/5) confirms the system is not hallucinating — all claims trace back to retrieved notes.

### Latency profile (warm cache, 5 queries × 5 runs)

| Stage          | Median | p95    |
|----------------|--------|--------|
| BM25 search    | 3.8ms  | 5.6ms  |
| Query embed    | 9.6ms  | 31.8ms |
| Vector search  | 2.5ms  | 4.2ms  |
| RRF fusion     | <1ms   | <1ms   |
| Graph expand   | <1ms   | <1ms   |
| **Full retrieve** | **16.9ms** | **20.9ms** |

Query embedding accounts for 57% of retrieval latency. Caching the sentence-transformer model reduced cold-path retrieval from **2.8s → 17ms (167× improvement)**.

---

## Setup

**1. Create the conda environment:**
```bash
conda create -n notes-agent python=3.11
conda activate notes-agent
pip install anthropic rank-bm25 lancedb sentence-transformers \
            beautifulsoup4 html2text fastmcp numpy
```

**2. Build the OCR binary** (requires Xcode command-line tools):
```bash
cd notes-agent
swiftc -O ocr_tool.swift -o ocr_tool -framework Vision -framework AppKit
```

**3. Store your Anthropic API key** (only needed for `--api` mode and summarization):
```bash
security add-generic-password -a "$USER" -s "ANTHROPIC_API_KEY" -w "sk-ant-..."
```

**4. Run the initial ingest:**
```bash
bash notes-agent/run_ingest.sh
```

First run exports all notes from Apple Notes, converts HTML → Markdown, OCRs image-only notes, chunks everything, embeds into LanceDB, and builds the similarity graph (~2–5 min depending on note count). Subsequent runs are incremental.

**5. (Optional) Auto-sync via LaunchAgent:**
```bash
cp notes-agent/com.notesagent.ingest.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.notesagent.ingest.plist
```
Runs a daily ingest automatically. Trigger manually anytime with `trigger_ingest` via MCP.

---

## Usage

**Ask a question:**
```bash
conda run -n notes-agent python notes-agent/agent.py "what are my project ideas?"
# Add --api to use Claude Haiku instead of local Ollama
conda run -n notes-agent python notes-agent/agent.py "what are my project ideas?" --api
```

**Interactive chat:**
```bash
conda run -n notes-agent python notes-agent/agent.py --chat --api
```

**Cross-note synthesis:**
```bash
conda run -n notes-agent python notes-agent/summarize.py "confidence and self belief"
```

**Raw retrieval (no LLM):**
```bash
conda run -n notes-agent python notes-agent/retrieve.py "project ideas" --top-k 5
```

**Run evals:**
```bash
conda run -n notes-agent python notes-agent/eval.py
conda run -n notes-agent python notes-agent/eval_judge.py --sample 20
```

**Latency profile:**
```bash
conda run -n notes-agent python notes-agent/benchmark.py --cold
```

---

## MCP (Claude Desktop / Claude Code)

Exposes 6 tools: `search_notes`, `get_note`, `list_recent_notes`, `browse_notes`, `summarize_topic`, `trigger_ingest`.

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "personal-notes": {
      "command": "/opt/miniconda3/envs/notes-agent/bin/python",
      "args": ["/path/to/notes-agent/mcp_server.py"]
    }
  }
}
```

**Claude Code** — add via CLI:
```bash
claude mcp add personal-notes \
  conda -- run -n notes-agent --no-capture-output \
  python /path/to/notes-agent/mcp_server.py
```

---

## Project structure

```
notes-agent/
├── run_ingest.sh        # orchestrates the full pipeline
├── export_notes.scpt    # AppleScript: export from Apple Notes
├── sync.py              # SHA256-based incremental change detection
├── convert_html.py      # HTML → Markdown conversion
├── ocr_notes.py         # Vision OCR for image-only notes
├── ocr_tool.swift       # Swift binary called by ocr_notes.py
├── chunk.py             # heading-aware semantic chunking
├── embed.py             # sentence-transformer embeddings → LanceDB
├── graph.py             # note-level similarity graph
├── retrieve.py          # hybrid BM25 + semantic + RRF + graph expansion
├── agent.py             # Q&A agent (Ollama / Claude Haiku)
├── summarize.py         # cross-note synthesis
├── mcp_server.py        # FastMCP server
├── eval.py              # retrieval eval (MRR, nDCG, P@k, R@k)
├── eval_judge.py        # LLM-as-judge answer quality eval
├── gen_eval.py          # auto-generate eval queries from corpus
├── benchmark.py         # per-stage latency profiler
└── utils.py             # shared helpers
```

---

## Notes on privacy

All embeddings, retrieval, and graph construction run locally. The only outbound calls are:
- `agent.py --api` → Claude Haiku (answer generation)
- `summarize.py` → Claude Haiku (synthesis)
- `eval_judge.py` → Claude Sonnet (judging) + Claude Haiku (generation)
- `gen_eval.py` → Claude Haiku (eval query generation)

Default mode (Ollama) makes no external calls.
