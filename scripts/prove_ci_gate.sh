#!/usr/bin/env bash
# Phase 16 exit criterion: "CI blocks a deliberately-regressed prompt."
#
#   ./scripts/prove_ci_gate.sh
#
# A gate nobody has watched fail is a green badge, not a safety net. This plants five real
# regressions — each the undoing of a fix that cost a live call, or the reopening of a
# disclosure — and asserts the offline suite goes red on every one, then puts the tree back.
#
# It runs ONE suite per scenario rather than the whole of run_e2e.sh, because a gate proof that
# takes ten minutes gets run once. Four scenarios live in the agent (reducer, prompt, tool gate,
# PHI logging) and need nothing but Python; the fifth is a cross-tenant database query and is
# SKIPPED, loudly, when no Postgres is reachable.
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

SKIPPED=0

# Runs one scenario: apply `patch_fn`, expect the named pytest selection to FAIL, then revert.
# `runner` is the service whose suite should catch it — "agent" (the default) or
# "scheduling_api", which needs the Postgres the e2e cluster provides.
scenario() {
    local name="$1" file="$2" target="$3" patch_fn="$4" runner="${5:-agent}"
    printf '\n=== %s\n' "$name"

    if [[ "$runner" == "scheduling_api" ]] && \
       ! pg_isready -h 127.0.0.1 -p "${CLINIC_E2E_PGPORT:-55432}" >/dev/null 2>&1; then
        echo "  ~ SKIPPED — no Postgres on :${CLINIC_E2E_PGPORT:-55432}; this gate is unproven."
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    TOUCHED+=("$file")
    "$patch_fn"

    # Always starts with `env`, never empty: under `set -u`, expanding an EMPTY array as
    # "${a[@]}" is an unbound-variable error on bash 3.2 (which is what macOS ships). That
    # error made the subshell exit non-zero, which this script reads as "the gate blocked it" —
    # so all four agent scenarios reported PASS without pytest ever running, and the "1 failed"
    # count came from the previous scenario's leftover log. A gate proof that can pass without
    # running the tests is worse than no gate proof.
    local env_prefix=(env)
    if [[ "$runner" == "scheduling_api" ]]; then
        env_prefix+=("CLINIC_DATABASE_URL=postgresql://postgres@127.0.0.1:${CLINIC_E2E_PGPORT:-55432}/clinic_test")
    fi
    : >/tmp/prove-gate.log   # never let a stale log supply the next scenario's failure count

    if (cd "$runner" && "${env_prefix[@]}" uv run pytest "$target" -q >/tmp/prove-gate.log 2>&1); then
        echo "  ✗ the gate stayed GREEN with the regression applied — it does not catch this."
        echo "    ran: $runner pytest $target"
        FAILED=$((FAILED + 1))
    else
        local n
        n="$(grep -Eo '[0-9]+ failed' /tmp/prove-gate.log | head -1)"
        echo "  ✓ blocked — $runner pytest $target: ${n:-failed}"
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

# 4. THE PHI BOUNDARY (Phase 17). Putting the caller's name back in the confirm-booking log line
#    is the defect that actually shipped and sat there for four phases, in a file where every
#    other sensitive field was carefully reduced to set/unset/len.
patch_phi_log() {
    perl -i -pe "s/\\Qredact('patient_name', args.get('patient_name'))\\E/args.get('patient_name')!r/" \
        agent/src/clinic_agent/scheduling_tools.py
}

# 5. TENANT ISOLATION (Phase 17). Dropping the clinic filter from the patient lookup in _verify
#    is the pre-Phase-17 query verbatim: with two clinics it reaches another tenant's chart
#    through the ordinary verification path.
patch_tenancy() {
    # Parameter ORDER is preserved (clinic_id, phone): the clinic_id binds to a tautology and
    # the phone still binds to the phone, so this is the old UNSCOPED lookup exactly, not a
    # broken query that would go red for the wrong reason.
    perl -i -pe 's/FROM patients WHERE clinic_id = %s AND phone = %s/FROM patients WHERE %s IS NOT NULL AND phone = %s/' \
        scheduling_api/app/db.py
}

scenario "prompt: the anti-fabrication rule is deleted" \
    agent/src/clinic_agent/prompts.py tests/test_prompt_contract.py patch_prompt

scenario "tool gate: VERIFICATION_REQUIRED_TOOLS is emptied" \
    agent/src/clinic_agent/scheduling_tools.py tests/ patch_gate

scenario "engine: the nudge stops respecting turn_had_tool" \
    agent/src/clinic_agent/core/reducer.py tests/test_reducer.py patch_nudge

scenario "PHI: the caller's name goes back into a log line" \
    agent/src/clinic_agent/scheduling_tools.py tests/test_phi.py patch_phi_log

scenario "tenancy: the patient lookup loses its clinic filter" \
    scheduling_api/app/db.py tests/test_tenancy.py patch_tenancy scheduling_api

printf '\n%s\n' "-----------------------------------------------------------"
if ((FAILED)); then
    echo "$PASSED/$((PASSED + FAILED)) regressions blocked — $FAILED SLIPPED THROUGH."
    echo "A regression the gate misses is a defect that reaches a caller. Add the missing test."
    exit 1
fi
if ((SKIPPED)); then
    echo "$PASSED/$PASSED regressions blocked, $SKIPPED SKIPPED (no database) — start Postgres"
    echo "to prove the tenant-isolation gate. See CLAUDE.md → Running the services."
    exit 0
fi
echo "$PASSED/$PASSED regressions blocked. The gate has teeth."
