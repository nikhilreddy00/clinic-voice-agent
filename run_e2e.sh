#!/usr/bin/env bash
# Everything a live call would exercise, without a phone: real reducer, real tool executor,
# real HTTP, real Postgres. Only the model and the microphone are scripted.
#
#   ./run_e2e.sh
#
# Brings up a THROWAWAY database and a scheduling API on its own port, runs every suite against
# them, and tears the API down. Never touches Supabase or anything on port 8000 — the e2e suite
# refuses to run unless the API reports the scratch database by name.
set -euo pipefail

PORT="${CLINIC_E2E_PORT:-8111}"
PGPORT="${CLINIC_E2E_PGPORT:-55432}"
DB="${CLINIC_E2E_DATABASE:-clinic_e2e}"
URL="postgresql://postgres@127.0.0.1:${PGPORT}/${DB}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! pg_isready -h 127.0.0.1 -p "$PGPORT" >/dev/null 2>&1; then
    echo "No Postgres on 127.0.0.1:${PGPORT}. Start one (see CLAUDE.md → Running the services)." >&2
    exit 1
fi
# clinic_test is the scheduling_api suite's own database. Created here too so a fresh machine
# (or a CI runner) needs no manual setup step that a developer would have done months ago.
createdb -h 127.0.0.1 -p "$PGPORT" -U postgres "$DB" 2>/dev/null || true
createdb -h 127.0.0.1 -p "$PGPORT" -U postgres clinic_test 2>/dev/null || true

echo "==> scheduling API on :${PORT} (database ${DB})"
( cd "$ROOT/scheduling_api" && CLINIC_DATABASE_URL="$URL" \
    uv run uvicorn app.main:app --port "$PORT" >/tmp/clinic-e2e-api.log 2>&1 ) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || {
    echo "API did not come up; see /tmp/clinic-e2e-api.log" >&2; exit 1; }

echo "==> scheduling_api"
( cd "$ROOT/scheduling_api" && CLINIC_DATABASE_URL="postgresql://postgres@127.0.0.1:${PGPORT}/clinic_test" \
    uv run pytest -q )

echo "==> agent (unit + end-to-end through the real API)"
( cd "$ROOT/agent" && CLINIC_E2E_API="http://127.0.0.1:${PORT}" \
    CLINIC_E2E_DATABASE_URL="$URL" CLINIC_E2E_DATABASE="$DB" uv run pytest -q )

echo "==> session_router"
( cd "$ROOT/session_router" && uv run pytest -q )

echo "==> tier 1 — recorded calls replayed through the reducer (no API calls, no cost)"
"$ROOT/agent/.venv/bin/python" "$ROOT/eval/tier1_replay.py"

echo "==> emergency detector (no API calls, no cost)"
"$ROOT/agent/.venv/bin/python" "$ROOT/eval/run_intent_eval.py" --detector-only

echo
echo "All suites green. Safe to place a live call."
