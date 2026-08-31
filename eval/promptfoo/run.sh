#!/usr/bin/env bash
# Run the per-intent eval suite.
#
#   ./run.sh                     # everything
#   ./run.sh tests/refill.yaml   # one intent, while iterating
#   ./run.sh --view              # open the browser report for the last run
#
# Two things this wrapper exists for, both of which fail confusingly otherwise:
#   * PROMPTFOO_PYTHON — the provider imports the agent package (and pipecat through it), so it
#     must run on the agent's venv, not whatever python3 is first on PATH.
#   * ANTHROPIC_API_KEY — the provider reads agent/.env itself, but promptfoo's own rubric
#     grader is a node process and needs the key in the environment.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

export PROMPTFOO_PYTHON="${PROMPTFOO_PYTHON:-$ROOT/agent/.venv/bin/python}"
[ -x "$PROMPTFOO_PYTHON" ] || { echo "no agent venv at $PROMPTFOO_PYTHON — run 'cd agent && uv sync'"; exit 1; }

if [ -f "$ROOT/agent/.env" ]; then
  export ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "$ROOT/agent/.env" | cut -d= -f2- | tr -d '"'"'"' ')"
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY is not set (agent/.env)"; exit 1; }

BIN=./node_modules/.bin/promptfoo
[ -x "$BIN" ] || BIN="npx --yes promptfoo@latest"

if [ "${1:-}" = "--view" ]; then exec $BIN view; fi

if [ $# -gt 0 ]; then
  # One suite: swap the tests list for just the file named.
  python3 - "$@" <<'PY' > .one-suite.yaml
import sys, pathlib, re
cfg = pathlib.Path("promptfooconfig.yaml").read_text()
cfg = re.sub(r"tests:\n(  - file://tests/.*\n)+",
             "tests:\n" + "".join(f"  - file://{a}\n" for a in sys.argv[1:]), cfg)
sys.stdout.write(cfg)
PY
  exec $BIN eval -c .one-suite.yaml --no-cache -j 6
fi

exec $BIN eval -c promptfooconfig.yaml --no-cache -j 6 --output results.json
