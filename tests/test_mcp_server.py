"""Exercise the FastMCP server over the in-memory MCP client transport."""

import asyncio
import json

import pytest
from fastmcp import Client

import mcp_server
from conftest import make_chunk

EXPECTED_TOOLS = {
    "search_notes", "get_note", "list_recent_notes",
    "browse_notes", "summarize_topic", "trigger_ingest",
}


def call(tool, args=None):
    async def go():
        async with Client(mcp_server.mcp) as c:
            return (await c.call_tool(tool, args or {})).data
    return asyncio.run(go())


def list_tools():
    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.list_tools()
    return asyncio.run(go())


def _schema(tool):
    return getattr(tool, "input_schema", None) or tool.inputSchema


@pytest.fixture
def index(tmp_path, monkeypatch):
    notes = [
        ("p1", "Sourdough journal", 1_700_000_000, "# Sourdough journal\n\nFed starter."),
        ("p2", "Gym plan", 1_750_000_000, "# Gym plan\n\nSquats."),
        ("p3", "Reading list", None, "# Reading list\n\nDune."),
    ]
    entries = []
    for nid, title, ts, body in notes:
        path = tmp_path / f"{nid}.md"
        path.write_text(body, encoding="utf-8")
        entries.append({"id": nid, "title": title, "modified_at": ts, "path": str(path)})
    idx = tmp_path / "_index.json"
    idx.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(mcp_server, "INDEX_FILE", idx)
    return entries


def test_all_six_tools_registered_with_schemas():
    tools = {t.name: t for t in list_tools()}
    assert set(tools) == EXPECTED_TOOLS

    search = _schema(tools["search_notes"])
    assert search["required"] == ["query"]
    assert search["properties"]["top_k"]["default"] == 6
    assert _schema(tools["get_note"])["required"] == ["note_id"]
    assert _schema(tools["summarize_topic"])["properties"]["max_notes"]["default"] == 15
    assert _schema(tools["trigger_ingest"])["properties"] == {}
    assert all(tools[n].description for n in EXPECTED_TOOLS)


def test_browse_notes_filters_by_keyword(index):
    out = call("browse_notes", {"keyword": "GYM"})
    assert "Notes matching 'GYM' (1 total)" in out
    assert "p2" in out and "p1" not in out
    assert "(3 total)" in call("browse_notes")


def test_get_note_returns_full_text_or_helpful_error(index):
    out = call("get_note", {"note_id": "p1"})
    assert out.startswith("# Sourdough journal\nID: p1")
    assert "Fed starter." in out
    assert "No note found with ID 'zzz'" in call("get_note", {"note_id": "zzz"})


def test_list_recent_notes_newest_first_and_skips_undated(index):
    out = call("list_recent_notes", {"limit": 10})
    assert "Most recent 2 notes (of 3 total)" in out
    assert out.index("Gym plan") < out.index("Sourdough journal")
    assert "Reading list" not in out


def test_tools_handle_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "INDEX_FILE", tmp_path / "missing.json")
    assert "No notes indexed yet" in call("browse_notes")
    assert "No notes indexed yet" in call("list_recent_notes")


def test_search_notes_formats_results_and_clamps_top_k(monkeypatch):
    import retrieve

    seen = {}
    long_text = "word " * 200

    def fake_retrieve(query, top_k):
        seen["args"] = (query, top_k)
        return [(make_chunk("p1", 0, long_text, title="Sourdough journal"), 0.0328)]

    monkeypatch.setattr(retrieve, "retrieve", fake_retrieve)
    out = call("search_notes", {"query": "bread", "top_k": 50})

    assert seen["args"] == ("bread", 12)
    assert "[1] Sourdough journal" in out and "id=p1" in out and "score=0.0328" in out
    assert "..." in out  # preview truncated at 600 chars
    assert "Unique notes: p1" in out

    monkeypatch.setattr(retrieve, "retrieve", lambda q, top_k: [])
    assert call("search_notes", {"query": "x"}) == "No relevant notes found for that query."


def test_summarize_topic_clamps_max_notes(monkeypatch):
    import summarize

    monkeypatch.setattr(summarize, "summarize", lambda t, max_notes: f"{t}:{max_notes}")
    assert call("summarize_topic", {"topic": "bread", "max_notes": 99}) == "bread:25"


def test_trigger_ingest_runs_script_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "AGENT_DIR", tmp_path)
    assert "Ingest script not found" in call("trigger_ingest")

    script = tmp_path / "run_ingest.sh"
    script.write_text("echo 'Changed/new: 2'\n")
    out = call("trigger_ingest")
    assert out.startswith("Ingest complete.") and "Changed/new: 2" in out

    script.write_text("echo boom >&2; exit 3\n")
    assert "Ingest exited with error (code 3)" in call("trigger_ingest")
