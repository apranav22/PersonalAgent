#!/bin/bash
# run_ingest.sh — daily ingest pipeline
# 1. Export from Apple Notes via AppleScript
# 2. Sync (compute changed/deleted IDs)
# 3. Convert changed notes HTML → Markdown

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$AGENT_DIR/ingest.log"

# log() must be defined before anything that can fail so errors are always recorded
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Trap any error and log it before exiting
trap 'log "ERROR: ingest failed at line $LINENO (exit $?)"' ERR

PYTHON="$(conda run -n notes-agent which python)" || {
    log "ERROR: could not find python in conda env 'notes-agent' — is the env created?"
    exit 1
}

log "=== Ingest start ==="

log "Exporting from Apple Notes..."
osascript "$AGENT_DIR/export_notes.scpt"

log "Running incremental sync..."
SYNC_OUTPUT=$($PYTHON "$AGENT_DIR/sync.py" --json)
echo "$SYNC_OUTPUT"

CHANGED=$(echo "$SYNC_OUTPUT" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(' '.join(d['changed']))")
DELETED=$(echo "$SYNC_OUTPUT" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(' '.join(d['deleted']))")

log "Converting HTML → Markdown..."
$PYTHON "$AGENT_DIR/convert_html.py"

log "Renaming notes to titles..."
$PYTHON "$AGENT_DIR/rename_notes.py"

log "OCR-ing image notes..."
$PYTHON "$AGENT_DIR/ocr_notes.py" ${CHANGED:+--ids $CHANGED}

log "Chunking changed notes..."
if [ -n "$CHANGED" ] || [ -n "$DELETED" ]; then
    $PYTHON "$AGENT_DIR/chunk.py" \
        ${CHANGED:+--changed $CHANGED} \
        ${DELETED:+--deleted $DELETED}

    log "Embedding changed chunks..."
    $PYTHON "$AGENT_DIR/embed.py" \
        ${CHANGED:+--changed $CHANGED} \
        ${DELETED:+--deleted $DELETED}

    log "Updating note graph..."
    $PYTHON "$AGENT_DIR/graph.py" \
        ${CHANGED:+--changed $CHANGED} \
        ${DELETED:+--deleted $DELETED}
else
    log "No changes — skipping chunk, embed, and graph steps"
fi

log "=== Ingest complete ==="
