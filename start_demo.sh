#!/usr/bin/env bash
#
# start_demo.sh — one command to bring up the whole demo on the host.
#
#   1. starts the mock scheduling API (with /metrics + /dashboard) on :8000
#   2. waits until it is healthy
#   3. prints the dashboard URL (and the public ngrok URL if ngrok is already running)
#   4. starts the voice agent in MODE=telephony (inbound LiveKit SIP calls to +14842951203)
#
# Then just call +14842951203. Ctrl-C stops everything; the agent prints its session summary
# (P50/P95/P99 latencies + call outcome) on the way out.
#
# ngrok is OPTIONAL and only needed to expose the dashboard/API publicly. The agent itself
# reaches LiveKit Cloud outbound and needs no inbound tunnel. To expose the dashboard:
#   ngrok http 8000     # in a separate terminal, before or after this script
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8000}"
API_PID=""
AGENT_PID=""

cleanup() {
  [[ -n "${_CLEANED:-}" ]] && return; _CLEANED=1
  echo
  echo "[demo] shutting down…"
  # Try SIGINT first (lets a foreground agent flush its session summary via its finally block),
  # then SIGTERM as a guaranteed stop — a backgrounded child ignores SIGINT, so TERM is what
  # actually reaps it. Per-call summaries are already flushed on caller hangup, so a hard stop
  # here loses nothing that matters.
  for pid in "$AGENT_PID" "$API_PID"; do
    [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "$AGENT_PID" "$API_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 1. scheduling API -----------------------------------------------------------------
echo "[demo] starting scheduling API on :$API_PORT …"
( cd "$ROOT/scheduling_api" && exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" ) &
API_PID=$!

# --- 2. wait for health ----------------------------------------------------------------
echo -n "[demo] waiting for API to be healthy"
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    echo " — up."
    break
  fi
  echo -n "."
  sleep 0.5
done
if ! curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
  echo
  echo "[demo] ERROR: scheduling API did not become healthy — check the log above." >&2
  exit 1
fi

# --- 3. dashboard + optional ngrok -----------------------------------------------------
echo "[demo] dashboard:  http://localhost:$API_PORT/dashboard/"
echo "[demo] metrics:    http://localhost:$API_PORT/metrics"
NGROK_URL="$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
  | grep -o 'https://[a-zA-Z0-9.-]*\.ngrok[a-zA-Z0-9.-]*' | head -n1 || true)"
if [[ -n "$NGROK_URL" ]]; then
  echo "[demo] public (ngrok): $NGROK_URL/dashboard/"
else
  echo "[demo] (ngrok not detected — run 'ngrok http $API_PORT' in another terminal for a public dashboard URL)"
fi

# --- 4. voice agent (telephony) --------------------------------------------------------
echo "[demo] starting voice agent (MODE=telephony) — call +14842951203. Ctrl-C to stop."
echo
# Background the agent (tracked PID) and wait on it, so the cleanup trap can deterministically
# stop BOTH services on Ctrl-C — including the programmatic case where only this script's shell
# is signalled. The agent still prints its per-call session summary on hangup and on shutdown.
( cd "$ROOT/agent" && MODE=telephony exec uv run python -m clinic_agent.pipeline ) &
AGENT_PID=$!
wait "$AGENT_PID"
