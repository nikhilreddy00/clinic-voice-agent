#!/usr/bin/env bash
# Phase 16 exit criterion: "CI blocks a deliberately-regressed prompt."
#
#   ./scripts/prove_ci_gate.sh
#
# A gate nobody has watched fail is a green badge, not a safety net. This plants three real
# regressions — one per layer, each the undoing of a fix that cost a live call — and asserts the
# offline suite goes red on every one, then puts the tree back.
#
# It runs the AGENT suite only, not the whole of run_e2e.sh: these three regressions live in the
# reducer, the prompt, and the tool gate, and none of them needs Postgres to prove. The full
# gate is what CI runs; this is what proves the gate has teeth.
#
# Refuses to start on a dirty tree. It edits tracked files in place and restores them from git,
# so uncommitted work would be at risk of being reverted along with the planted damage.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is dirty — commit or stash first (this script reverts tracked files)" >&2
    exit 2
fi

TOUCHED=()
restore() {
    if ((${#TOUCHED[@]})); then
        git checkout -- "${TOUCHED[@]}" 2>/dev/null || true
    fi
}
trap restore EXIT

PASSED=0
FAILED=0

# Runs one scenario: apply `patch_fn`, expect the named pytest selection to FAIL, then revert.
scenario() {
    local name="$1" file="$2" target="$3" patch_fn="$4"
    printf '\n=== %s\n' "$name"

    TOUCHED+=("$file")
    "$patch_fn"

    if (cd agent && uv run pytest "$target" -q >/tmp/prove-gate.log 2>&1); then
        echo "  ✗ the gate stayed GREEN with the regression applied — it does not catch this."
        echo "    ran: agent pytest $target"
        FAILED=$((FAILED + 1))
    else
        local n
        n="$(grep -Eo '[0-9]+ failed' /tmp/prove-gate.log | head -1)"
        echo "  ✓ blocked — agent pytest $target: ${n:-failed}"
        PASSED=$((PASSED + 1))
    fi

    git checkout -- "$file"
}

# 1. THE PROMPT. The anti-fabrication rule is the one sentence between a caller and hanging up
#    believing they have an appointment that does not exist. Deleting it is the regression the
#    phase exit criterion names.
patch_prompt() {
    # perl, not sed: `sed -i ''` is BSD-only and this has to run on a Linux runner too.
    perl -i -ne 'print unless /^NEVER CLAIM AN ACTION YOU DID NOT TAKE/' \
        agent/src/clinic_agent/prompts.py
}

# 2. THE TOOL GATE. Emptying the verification set is how a PHI lookup would quietly become an
#    HTTP request for a caller who has proved nothing. The gate is in the reducer, so Tier 1
#    sees it as an action that should never have existed.
patch_gate() {
    printf '\n# regression planted by scripts/prove_ci_gate.sh\nVERIFICATION_REQUIRED_TOOLS = frozenset()\n' \
        >> agent/src/clinic_agent/scheduling_tools.py
}

# 3. THE ENGINE. Dropping `turn_had_tool` from the nudge condition is the 2026-09-03 defect
#    verbatim: a successful hold's read-back gets re-prompted, which closes the live TTS context
#    and truncates the confirmation 1 s into an 8 s sentence.
patch_nudge() {
    perl -i -pe 's/^        and not state\.turn_had_tool$/        and True  # regression/' \
        agent/src/clinic_agent/core/reducer.py
}

scenario "prompt: the anti-fabrication rule is deleted" \
    agent/src/clinic_agent/prompts.py tests/test_prompt_contract.py patch_prompt

scenario "tool gate: VERIFICATION_REQUIRED_TOOLS is emptied" \
    agent/src/clinic_agent/scheduling_tools.py tests/ patch_gate

scenario "engine: the nudge stops respecting turn_had_tool" \
    agent/src/clinic_agent/core/reducer.py tests/test_reducer.py patch_nudge

printf '\n%s\n' "-----------------------------------------------------------"
if ((FAILED)); then
    echo "$PASSED/$((PASSED + FAILED)) regressions blocked — $FAILED SLIPPED THROUGH."
    echo "A regression the gate misses is a defect that reaches a caller. Add the missing test."
    exit 1
fi
echo "$PASSED/$PASSED regressions blocked. The gate has teeth."
