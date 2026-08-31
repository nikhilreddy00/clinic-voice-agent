# CLAUDE.md — Clinic Voice Agent

Guidance for AI assistants (and humans) working in this repo. Read this first so you don't
re-derive project context each session.

## Project purpose

A production-shaped **inbound appointment-scheduling voice agent** for a healthcare clinic,
built as a portfolio project. A caller phones the clinic; the agent greets them, discloses it
is an AI, collects their intent and details, offers open appointment slots, confirms a choice,
books it, and closes the call. The emphasis is on the *shape* of a real system — dialogue
design, tool calls, observability, evals, and governance — not on production PHI handling.

**All patient data is synthetic. Treat PHI as sensitive: no real patient data ever enters
this repo, logs, or prompts.**

## Architecture — tech stack per layer

| Layer | Responsibility | Chosen tech |
|-------|----------------|-------------|
| Telephony (later) | Inbound PSTN/SIP call ingress | LiveKit SIP (free tier, one number included: `+14842950169`) |
| Transport | Real-time audio in/out | Pipecat transport (WebRTC local dev → SIP later) |
| ASR | Speech → text | Deepgram (`pipecat.services.deepgram`) |
| LLM | Reasoning / dialogue | Anthropic Claude Haiku 4.5 (`pipecat.services.anthropic`) — see LLM-provider note below |
| TTS | Text → speech | Cartesia (`pipecat.services.cartesia`); Piper for free-tier testing (TODO) |
| Orchestration | Pipeline, turn-taking, context | **In-house event loop** (`clinic_agent.core`, Phase 10); Pipecat retained as the fallback path |
| Scheduling backend | Availability / hold / booking | FastAPI service (`scheduling_api/`) |
| Storage | Slots & bookings | **Postgres** (`CLINIC_DATABASE_URL`) — Phase 9; was SQLite |

### LLM-provider note (why Claude, not Groq)

The dialogue LLM was **Groq-hosted Llama** through Phase 2, then swapped to **Anthropic Claude
Haiku 4.5** during Phase 3 to unblock eval iteration: Groq's free tier has a 100K-tokens/day cap
that a full 16-case eval run exhausts on its own, which stalled Phase-3 development. Claude Haiku
4.5 is the fastest/cheapest Claude model and reasonable for real-time voice latency.

- **Groq remains available as a dormant option for future latency benchmarking.** The `groq`
  Pipecat extra is still installed and `GROQ_API_KEY`/`GROQ_MODEL` still load in `config.py`, but
  nothing is wired to them. **Switching back to Groq is a deliberate code change in
  `agent/src/clinic_agent/pipeline.py` and `eval/run_eval.py`** — there is no runtime provider
  flag. This keeps the active path simple while leaving the fallback one edit away.
- This is a **development-unblocking swap, not a permanent architecture decision.** Groq's low
  latency may still make it the better choice for the final production/demo voice build.

## Dialogue states (summary)

`GREETING_DISCLOSURE → COLLECT_INTENT → COLLECT_NAME → COLLECT_REASON → OFFER_SLOTS →
CONFIRM_SLOT → BOOK → CLOSE`, plus global `FALLBACK` / `NO_MATCH` and `ESCALATE_HUMAN`.

The greeting includes the AI disclosure (and, on telephony, a call-recording consent line).
Full state machine — data collected, validation, transitions, and fallback behavior per
state — is in [`docs/build_spec.md`](docs/build_spec.md).

## Folder structure

```
.
├── CLAUDE.md                  # this file
├── .claudeignore              # excludes for AI tooling
├── docs/build_spec.md         # dialogue state machine spec
├── agent/                     # Pipecat voice-agent service (SKELETON until Phase 1)
│   ├── pyproject.toml         # deps pinned; NOT installed in Phase 0
│   └── src/clinic_agent/
│       ├── pipeline.py        # TODO-marked pipeline skeleton (no logic yet)
│       ├── config.py          # env/settings loading
│       └── prompts.py         # system-prompt placeholder + AI-disclosure text
└── scheduling_api/            # FastAPI mock backend (WORKING)
    ├── pyproject.toml
    ├── app/{main,models,db,seed_data}.py
    └── tests/test_api.py
```

The agent and the scheduling API are **separate services / processes** with their own
`pyproject.toml` (managed with `uv`).

## Build plan

This reflects the *actual* build sequence. (Earlier drafts of this file had an 8-phase plan
that listed the eval harness and barge-in as separate later phases; both landed earlier than
planned — the eval harness in Phase 3, barge-in in Phase 4 — so the remaining work is
renumbered to what's actually left.)

**Completed:**

- **Phase 0 — Scaffold, docs, mock scheduling API.** ✅
- **Phase 1 — Live ASR→LLM→TTS loop.** Local/WebRTC transport, greeting/AI disclosure, mic
  gate, core dialogue happy path. ✅
- **Phase 2 — LLM-driven booking via tool calls.** Wire the agent to the scheduling API
  (availability / hold / confirm). ✅
- **Phase 3 — Headless eval harness.** 16 scripted + adversarial conversations, structural
  (tool-trace) scoring. ✅
- **Phase 4 — Dialogue/API defect fixes + barge-in.** Five target fixes (see
  `docs/build_spec.md`) + turn-taking / interruption handling with a false-positive metric. ✅
- **Phase 5 — Telephony.** Real inbound phone number via LiveKit SIP (free tier,
  `+14842950169`), `MODE=local|telephony` runtime switch, idempotent SIP trunk + Direct-dispatch
  setup script (`agent/scripts/setup_livekit_sip.py` → room `clinic-inbound`), AI disclosure +
  call-recording consent in the greeting on the telephony path. **Verified end to end** —
  multiple live inbound calls connected and booked appointments. ✅
- **Phase 6 — Observability + Docker + demo.** Per-turn latency (P50/P95/P99), ASR confidence,
  tool-success + outcome metrics via a structured JSONL sink (`logs/calls.jsonl`) alongside the
  console logs; `GET /metrics` + a `/dashboard` on the scheduling API; both services Dockerized
  (`docker compose up`) with a shared logs volume; `start_demo.sh` one-command host launcher +
  ngrok for public dashboard access. **Real numbers captured on a live call** (see
  `docs/build_spec.md` → *Phase 6*). No cloud deployment — the demo runs the agent locally
  (outbound to LiveKit) and uses ngrok only to expose the dashboard. ✅

- **Phase 7 — README + demo.** Recruiter-facing `README.md` (Try-it phone number up top, ASCII
  architecture diagram, tech-stack + latency + eval tables, five "production-shaped" trust factors,
  resume line), `docs/demo_script.md` (three-scenario recording guide), and final repo hygiene
  (stale scaffolding TODOs resolved, gitignore/secret checks). Nothing new built — documents the
  existing system accurately. ✅

**Phases 0–7 complete.** The project then entered a **production-scale rebuild (Phases 8–17)** —
see `/Users/uvnikhil/.claude/plans/cheerful-enchanting-comet.md` for the full plan. In progress:

- **Phase 8 — model bake-off.** `eval/providers.py` (backend abstraction over Anthropic + the
  OpenAI wire format for Groq/Cerebras, all streaming so TTFT is measurable) and `eval/bakeoff.py`
  (runs the 19-case suite per candidate; compares TTFT, cost/call, and cache viability). Harness
  done; the full sweep has not been run. **Key measurement: prompt caching cannot engage on Haiku
  4.5** — the cacheable prefix (tools + system) is 3,811 tokens against a 4,096 minimum, and
  Anthropic accepts `cache_control` then silently caches nothing. Sonnet 4.6/5 cache fine
  (1,024 minimum). Do NOT "shrink the prompt" — that makes it permanently impossible.
- **Phase 9 — data layer.** ✅ SQLite → Postgres. Every state transition is a single-statement
  compare-and-swap (`UPDATE ... WHERE <expected state> RETURNING`), which is what makes concurrent
  holds safe: the old read-then-write let **9 of 20** concurrent callers "win" the same slot.
  Adds idempotency keys (a retried booking replays the original confirmation instead of creating
  a second appointment), a background sweeper so `/availability` performs no writes, optional
  `CLINIC_API_TOKEN` auth, and the Phase 12/13 schema landed early so there is one migration.
  Fixed the seed timezone bug: slots were built as 09:00 **UTC** (05:00 clinic-local); they are
  now generated in `America/New_York` and stored as `timestamptz`.
- **Phase 10 — in-house event loop.** ✅ New package `agent/src/clinic_agent/core/` replaces
  Pipecat's `Pipeline`/`FrameProcessor` chain with `reduce(state, event) -> (state, actions)` —
  pure, no I/O, no clock reads, ids derived from state counters. Adapters (media/turn/STT/LLM/
  TTS/tools) talk to the vendors directly and communicate only in events. **Media transport is
  still bought** (LiveKit SIP / local audio); Silero VAD is kept from Pipecat as a library.
  Full detail: `docs/build_spec.md` → *Phase 10*. Things to know before touching this:
  - **Raw audio is never an event** — `media → TurnEngine` emits only derived decisions. That's
    what makes traces committable and replay exact.
  - **`python -m clinic_agent.core` is the new entrypoint**; `clinic_agent.pipeline` (Pipecat)
    is unchanged and still runs. The new engine is **not yet the default** — it has not carried
    real audio on a live call.
  - **All per-call state is constructed in `CallSession`**, fixing the Phase-1..9 defect where
    it lived in `run_agent()` locals and caller #2 inherited caller #1's history.
  - **`scheduling_tools.execute_tool()` is shared by both engines.** Change tool behavior there,
    never in one path only.
  - Adapters are built via overridable `_build_*` methods on `CallSession` — that seam is what
    Phase 11's Tier-A load test (fake adapters, injected latency) uses.
- **Phase 12 — reasoning layer.** ✅ `core/intent.py` (deterministic emergency detector + the
  classifier contract), `core/llm_router.py` (pure `select_tier` + tier→model resolution),
  intent-scoped prompts (`prompts.build_system_prompt`) and tool subsets
  (`build_tools_schema(intent)`), `core/adapters/classifier.py`, and
  `eval/run_intent_eval.py`. Measured: **emergency recall 100%** (61 cases, 0 false positives),
  **intent accuracy 98.3%**, non-scheduling prompts **−77%** (11.9 KB → ~2.7 KB; was ~1.9 KB
  until the unconditional anti-fabrication rule moved into the core prompt). Full detail:
  `docs/build_spec.md` → *Phase 12*. Rules that are easy to break:
  - **The emergency path never touches a model.** `detect_emergency` is pure and runs in the
    reducer before any request exists. Do not "improve" it by asking an LLM — that makes the
    highest-stakes control probabilistic and kills replayability.
  - **Never apply a general negation rule to emergency phrases.** "can't breathe" / "not
    breathing" carry their own polarity; the narrow guard applies ONLY to `NEGATABLE_PATTERNS`.
  - **Intent is sticky** (`intents.resolve_intent`), and stickiness is a SAFETY control, not a
    tidiness one. Two live regressions, both ending in "zero tool calls, outcome abandoned":
    mid-booking slot-fill answers classify as `unknown`, and — the expensive one — the answer to
    the agent's own "what's the reason for your visit?" classifies as `clinical_question` at
    high confidence, because a clinical description is what that question ASKS FOR. Either one
    overwriting the intent strips the scheduling tools *and* swaps the booking prompt for a
    hand-off fragment, after which the model narrates a booking it never made, confirmation
    number included. No confidence threshold can separate the two (0.85 was tried; it failed at
    0.92) — hence `QUESTION_INTENTS`: from a scheduling flow, questions never preempt, only a
    genuine change of task does.
  - Classifier latency is **928 ms p50, not the planned <150 ms** — Haiku is not a fast-tier
    model. It costs the caller nothing (it runs in parallel) but scoping applies from the next
    turn. `CLINIC_MODEL_FAST` points the tier elsewhere once Phase 8's bake-off runs.
- **Phase 11 — concurrency + load proof.** ✅ `core/worker.py` (N sessions per process, prewarmed
  analyzer pool, graceful drain), `core/router_client.py`, `session_router/` (LiveKit
  `room_started` webhook → least-loaded worker, heartbeat expiry, orphan re-dispatch), and
  `loadtest/` (Tier-A harness + SVG chart). Measured on one process: **1,000 concurrent sessions
  loop-mode and 1,600 with real VAD on every frame, with flat p50/p95 and zero timeouts**;
  event-loop lag starts growing superlinearly past ~400, so **capacity should be set near 800**.
  Full numbers + caveats: `docs/build_spec.md` → *Phase 11*. Notes:
  - **The latency knee was never reached** — do not quote one. What moves is loop lag.
  - **Tier B (real concurrent calls) is NOT run** — no cost-per-minute number exists yet.
  - The router's LiveKit webhook is **unauthenticated**; verify the signing JWT before deploying
    it anywhere public.
- **Phase 13 — agent memory + expanded tool surface.** ✅ Caller memory by ANI, a hard identity
  gate, and six new tools (`verify_identity`, `list_appointments`, `reschedule_appointment`,
  `cancel_appointment`, `request_refill`, `get_clinic_info`) across `scheduling_api` (7 new
  endpoints, `db.py` Phase-13 section) and the agent (`scheduling_tools.py`, `core/reducer.py`,
  `prompts.caller_context_note`). Full detail: `docs/build_spec.md` → *Phase 13*. The rules that
  are easy to break, all of them downstream of one fact — **caller ID is spoofable**:
  - **The gate is in the reducer, not the prompt.** A tool in `VERIFICATION_REQUIRED_TOOLS`
    never becomes an `InvokeTool` action while `state.identity_verified` is false, so an
    unverified request never exists on the wire. The refusal MUST still be handed back as a
    `tool_result` — a refused tool with no result means no `ToolCompleted`, which means the turn
    never drains and the call sits in silence.
  - **The API re-verifies `(phone, date_of_birth)` on every single PHI call anyway.** The flag
    is a UX gate in a process driven by a language model; the database is the security boundary.
    Do not "optimize" this into a session or a verification token.
  - **A wrong DOB and an unknown number must return the identical 403 body.** Differentiating
    them turns the endpoint into a patient-enumeration oracle.
  - **Never speak a caller's name before verification** — not in the greeting, not in the
    context note. It is the same disclosure the gate exists to prevent, one turn earlier. A
    recognised caller gets "you've reached us before" and nothing identifying;
    `/caller-memory` deliberately returns no name for the same reason.
  - **`phone` and `date_of_birth` are injected by the reducer and OVERWRITE the model's
    values** on caller-scoped tools, so a number in a transcript cannot redirect a lookup.
    `confirm_booking` is the one exception on DOB (there it is intake, not a credential) — and
    booking with a phone + DOB is what creates the patient record that makes the next call a
    returning one.
  - **The agent never approves a refill.** `request_refill` creates a `staff_tasks` row and
    there is no code path that could do otherwise.
  - Memory is loaded **after** the greeting action and never awaited: the AI disclosure must not
    wait on a database, and a failed lookup just means a colder greeting.

**Per-intent hardening + promptfoo evals (post-Phase-13).** `eval/promptfoo/` — 72 behavioural
cases, one suite per intent, **100% on three consecutive runs**. Full detail:
`docs/build_spec.md` → *Per-intent evals*. The rules it produced, all of them load-bearing:
  - **The classifier may never set `Intent.EMERGENCY`** (`intents.CLASSIFIER_ONLY_ADVISORY`).
    `detect_emergency` owns that path. A live call classified a knee laceration as emergency at
    0.95 twice; the label strips every tool and swaps the prompt while the scripted 911 path
    stays untouched, so the model spent 100 seconds saying "I'm booking you right now" with
    nothing to call, invented a `book_appointment` tool, and read its own `<thinking>` block —
    hold UUID included — to the caller.
  - **`<thinking>` never reaches TTS** (`reducer._strip_thinking`). Held across streaming
    deltas, dropped entirely if never closed. Everything the reducer treats as text is spoken.
  - **`reducer._ACTION_CLAIM` — one follow-through nudge per caller turn.** A turn that
    announces an action and calls no tool is re-prompted once. This is engine enforcement
    because the prompt rule (with a worked example) demonstrably did not hold.
  - **`media.end_utterance` puts a MARKER on the queue; it must not decide completion itself.**
    The old `if self._speaking is None` test lost a race on short replies, so
    `BotStoppedSpeaking` never fired, the mic gate never reopened, and the agent went deaf for
    the rest of the call. Plus a 5 s stall timeout for a provider that never sends `done`.
  - **`intent._is_recollection`** — narrowly scoped to "passed out"/"blacked out"/"fainted",
    requiring EXPLICIT past-tense evidence. Never widen it: absence of urgency cues is not
    evidence, and a general rule here would suppress "can't breathe" the way a general negation
    rule would.
  - The eval provider imports the agent's real prompt/tool builders and mirrors the nudge, so
    it measures the system a caller meets. Do not let it grow a private copy of the prompt.

## Conventions & governance

- **Synthetic data only.** No real PHI in code, seeds, logs, or prompts.
- **Secrets** live in each service's local `.env` (git-ignored), never committed. Real API
  keys must never enter tracked files.
- **AI disclosure** is mandatory in the greeting; call-recording consent is required once
  telephony lands (Phase 7). See `agent/src/clinic_agent/prompts.py`.
- Keep the agent and scheduling API decoupled — the agent talks to the API over HTTP.
- **Concurrency: N sessions per process (superseded rule).** This file used to say one
  container per voice session was mandatory, because the GIL means CPU-bound audio/VAD work on
  one call would stall the others. **The mechanism was real; the conclusion was wrong**, and
  Phase 11 measured it. The fix is making per-session work non-blocking, not giving each caller
  a process:
  - shared Silero ONNX weights + a bounded inference pool (`core/vad.py`) — **7.97 MB → 0.002 MB
    and 23.4 ms → 0.46 ms per session**, bit-identical output;
  - buffered/shared JSONL sinks (`core/jsonl.py`) instead of `open/write/close` per event on the
    loop (22.4 µs → 0.9 µs per record);
  - vectorized `frame_rms` (Phase 10), which runs 50×/second/session.

  `core/worker.Worker` hosts N `CallSession`s with a prewarmed pool; `loadtest/` drives 1,000
  concurrent sessions in one process. See `docs/build_spec.md` → *Phase 11* for the measured
  curve and the knee.

  **What still pins production to one replica is the SIP dispatch rule, not the engine.** Direct
  dispatch puts every caller in one shared room, so a second replica would talk over the first.
  Room-per-call requires `scripts/setup_livekit_sip.py --dispatch individual` **plus** the
  session router and registered workers — and it takes the live demo number off the air until
  all three are running, so it is a deliberate operator step (see `agent/railway.toml`).

## Running the services

```bash
# Scheduling API — needs Postgres (Phase 9). Any throwaway database will do:
#   initdb -D /tmp/pgclinic -U postgres --auth=trust
#   pg_ctl -D /tmp/pgclinic -o "-p 55432 -c listen_addresses=127.0.0.1 \
#       -c unix_socket_directories=''" -l /tmp/pgclinic.log start
#   createdb -h 127.0.0.1 -p 55432 -U postgres clinic_dev   # and clinic_test, clinic_eval
export CLINIC_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/clinic_dev
cd scheduling_api && uv sync --extra dev && uv run uvicorn app.main:app --reload
cd scheduling_api && uv run pytest        # 70 tests; SKIPPED if no Postgres is reachable

# --- Supabase is the backend database ------------------------------------------------
# Project : clinic-voice-agent   ref qhrvyhssfytkrfcodbuc   region us-east-1   Postgres 17.6
# Schema + seed are already applied (11 tables, RLS on all, 36 slots).
#
# Connection: use the SESSION POOLER. Verified on this project:
#     aws-0-us-east-1.pooler.supabase.com  -> IPv4   (works)
#     db.qhrvyhssfytkrfcodbuc.supabase.co  -> IPv6 ONLY (hangs on an IPv4 network)
#
export CLINIC_DATABASE_URL='postgresql://postgres.qhrvyhssfytkrfcodbuc:<DB_PASSWORD>@aws-0-us-east-1.pooler.supabase.com:5432/postgres'
#
# Get <DB_PASSWORD> from Dashboard -> Project Settings -> Database. It is shown once at
# creation; if you don't have it, use "Reset database password" there.
#
# Apply + verify the schema against any database (idempotent):
cd scheduling_api && uv run python scripts/migrate.py
#
# Gotchas, all handled in code but worth knowing:
#   * Transaction pooler (port 6543) does NOT support prepared statements, and psycopg3 starts
#     preparing after 5 executions -- it works, then breaks on the 6th. db.py auto-detects 6543
#     and disables preparation (override: CLINIC_DB_PREPARE_THRESHOLD).
#   * RLS is enabled deny-by-default on every table. Supabase exposes `public` via its Data API
#     and bookings/patients hold PHI, so without it they'd be readable with the browser-side
#     publishable key. `supabase db advisors` reports 11 INFO "RLS enabled, no policy" -- that
#     is the INTENDED posture here, not a defect. Do NOT add permissive policies to silence it.
#   * FREE TIER PAUSES after ~7 days without database activity. A paused project means the
#     agent cannot book. Keep it warm, or upgrade before relying on the demo number.

# Voice agent — in-house event loop (Phase 10). MODE=local | telephony
cd agent && uv run python -m clinic_agent.core
cd agent && MODE=telephony uv run python -m clinic_agent.core

# Voice agent — Pipecat path (unchanged fallback; same MODE switch)
cd agent && uv run python -m clinic_agent.pipeline

# Telephony either way needs the one-time, idempotent SIP trunk + dispatch rule:
cd agent && uv run python scripts/setup_livekit_sip.py

cd agent && uv run pytest        # 330 tests, no network/keys needed

# Phase-12 intent eval. --detector-only runs the SAFETY half with no API calls and no cost.
agent/.venv/bin/python eval/run_intent_eval.py --detector-only
agent/.venv/bin/python eval/run_intent_eval.py

# Per-intent behavioural evals (promptfoo). One suite per intent; 72 cases; needs an API key.
cd eval/promptfoo && ./run.sh                    # everything
cd eval/promptfoo && ./run.sh tests/refill.yaml  # one intent, while iterating
cd eval/promptfoo && ./run.sh --view             # browse the last run

# Session router (Phase 11 control plane) — only needed for room-per-call dispatch
cd session_router && uv sync --extra dev && uv run pytest    # 26 tests
cd session_router && uv run uvicorn app:app --port 8080

# Tier-A load test (no keys, no network — synthetic providers, real orchestrator)
agent/.venv/bin/python loadtest/tier_a.py --mode loop  --concurrency 1,10,100,1000 --turns 6
agent/.venv/bin/python loadtest/tier_a.py --mode audio --concurrency 5,100,800,1600 --turns 4
agent/.venv/bin/python loadtest/tier_a.py --rechart loadtest/results/tierA-loop.json
```

```bash
# Landing page — no build step, no server needed. Open it, or check its invariants.
open site/index.html
python3 site/check_page.py                                   # asserts every published claim
python3 site/build_trace.py logs/traces/<call_id>.jsonl       # re-point the replay at a new call
```

`MODE` (default `local`) is the only switch between the laptop mic/speaker path and the LiveKit
SIP telephony path — the ASR→LLM→TTS loop, tool calls, mic gate, and barge-in are identical in
both. Full telephony setup steps: `docs/build_spec.md` → *Phase 5 — Telephony (LiveKit SIP)*.

## Current status

**Phases 0–7 shipped the working product; the production-scale rebuild is at Phase 13 of 17.**
Phase 8 (model bake-off harness) is built but the sweep has not been run; Phases 9 (Postgres +
Supabase), 10 (in-house event loop), 11 (concurrency + load proof), 12 (reasoning layer), and 13
(memory + verified tool surface) are done, and the engine is validated by real phone calls (see
below). **Next: Phase 14 — latency finish** (semantic EOU, speculative LLM start,
sentence-boundary TTS chunking, connection pooling, region colocation).

**Phase 13's exit criterion is not met yet, and it needs a phone, not code:** a returning caller
recognized, verified, and rescheduling end to end on a live call. Everything below the
microphone is proven — 70 API tests including the spoofed-ANI adversarial set, 299 agent tests,
and a full book → recognise → refuse → verify → list → reschedule → refill → cancel run against
the real API and Postgres.

**Live-call hardening (post-Phase-12, effectively early Phase 14).** Five defects, all found by
reading traces rather than by a failing test. Do not regress these:
  - **Deepgram closes an idle socket after 10 s** (`net0001`). `MODE=telephony` opens STT at
    start-up and then waits an unbounded time for a call, and the mic gate sends nothing while
    the bot speaks — so both windows killed it. `stt.py` sends a KeepAlive on an interval.
    Nothing reconnects, so a dead socket means a call that hears nothing and looks fine.
  - **`intents.QUESTION_INTENTS`** — see the Phase-12 note. This is the fabricated-booking fix.
  - **`core/endpointing.py` — graded turn-end grace.** Deepgram ends a turn after 300 ms of
    silence, so a caller pausing to think gets talked over; 7 of 27 turns on one call were the
    agent answering a fragment. Grace is graded ON PURPOSE: a trailing comma or a word that
    cannot end a sentence buys 1.6 s, merely-unpunctuated buys 0.7 s. That split is load-bearing
    — "December eight two thousand" is a complete answer with no full stop, and taxing it 1.6 s
    would slow every name and date. Windows renew up to 3×. **This deliberately raises per-turn
    e2e** (~800 ms) while halving total call time; do not "optimize" it away by reading the
    per-turn number alone.
  - **`core/closing.py` — the agent can end a call.** `EndCall` used to have exactly one
    producer, the caller hanging up, so a completed booking left the line open until the caller
    gave up (16 s of dead air, measured). Gated on `state.booked` because hanging up is
    irreversible, and it fires on `BotStoppedSpeaking`, not on the farewell — the caller has to
    hear the goodbye.
  - **The greeting must wait for media, not signalling.** `CallerPresent` fired on LiveKit's
    `participant_connected`; the agent began speaking 99 ms later, into a track with no
    receiver, and callers reported the greeting as broken. It now fires on `track_subscribed`
    plus `CLINIC_GREETING_SETTLE_MS` (400 ms) — a calibration knob for carrier variance, not
    superstition.
  - **A Cartesia error frame on a cancelled context is NOT provider degradation.** Every
    barge-in produced one; `degraded` drives the Phase-15 breaker, so it must not be poisoned by
    normal turn-taking.

Two open items carried forward, both requiring real-world execution rather than code:
- ~~No live call through the in-house engine.~~ **CLOSED — 5 calls placed, 3 booked end to end**,
  verified in Postgres. Measured across the booked calls: **voice-to-voice p50 1.5 s** (vs 4.1 s
  on the Pipecat path) and **call duration halved, ~6 min → ~3 min**. `core` is now the proven
  path; making it the *default* entrypoint is a one-line change nobody has made yet.

  **Read `logs/traces/<call_id>.jsonl` before trusting any call.** Every defect below was found
  in the event stream and was invisible in the console logs — a call can sound flawless and have
  booked nothing. `had_tool_call: false` on every turn is the tell.
- **Tier B load (25–50 real concurrent calls) is not run**, so there is no cost-per-minute
  number. The harness and control plane exist; the spend and PSTN capacity do not.

The original Phase 0–7 record follows. Phases 0–5 are done (scaffold + mock API, live
ASR→LLM→TTS loop, scheduling-API tool calls, dialogue hardening + headless eval, barge-in, and
LiveKit SIP telephony verified end to end by live inbound calls). Phase 6 added observability,
Docker, and the demo launcher; Phase 7 shipped the recruiter-facing `README.md`,
`docs/demo_script.md`, and final repo hygiene. Phase 6 detail:

- **Per-turn latency + outcomes** (`agent/src/clinic_agent/metrics.py`): a `LatencyCollector` fed
  by three pass-through `MetricsTap` processors (after STT/LLM/TTS) writes a structured JSONL sink
  (`logs/calls.jsonl`) *alongside* the console logs — ASR/LLM/TTS/E2E latency, ASR confidence,
  per-tool outcomes, and a per-call P50/P95/P99 summary (also printed as a human `[session]` line
  on hangup). The VAD-silence boundary is `VADUserStoppedSpeakingFrame` (NOT
  `UserStoppedSpeakingFrame` — the wrong class silently records `turns=0`; guarded by a canary
  test). Idle-timeout is disabled on the telephony path so the agent waits for an inbound call.
- **Dashboard**: `GET /metrics` aggregates the log; `/dashboard` (served same-origin, vanilla JS)
  shows the percentile table, outcomes, ASR confidence, tool success, and last-5 calls.
- **Docker + demo**: `agent/Dockerfile`, `scheduling_api/Dockerfile`, `docker-compose.yml`
  (shared logs volume, secrets via `env_file`), and `start_demo.sh` (host launcher: API + agent
  in `MODE=telephony` + optional ngrok). **Live-call numbers captured** (E2E p50 ≈ 4.1 s, LLM the
  dominant stage — see `docs/build_spec.md` → *Phase 6*). Docker files are validated but not
  locally built (Docker not installed on the dev machine).

**Earlier phases (recap).** Phase 4 fixed all five confirmed defects from the Phase-4 target list
in [`docs/build_spec.md`](docs/build_spec.md) and added barge-in:

- **Five targets fixed** (date-grounding via an injected date→weekday table; past-time slot
  filter in `scheduling_api`; mid-flow slot tracking + single confirmation gate; empty-window
  fallback; past-time read-back guard), **plus** a bonus fix for a fabricated-booking bug (agent
  read a `hold_id` back as a confirmation number without calling `confirm_booking`). Eval moved
  from **88% task completion / 81% overall pass → 100% / 100%** (16/16, slot-filling 8/9 → 11/11).
- **Barge-in / turn-taking** (`agent/src/clinic_agent/barge_in.py` + `pipeline.py`): Pipecat's
  built-in interruption, gated by `MinWordsUserTurnStartStrategy(min_words=3)` at the turn level
  and a sustained-speech mic gate (~600 ms threshold, 400 ms echo-tail hangover) at the audio
  level, with a false-positive-interruption metric. Thresholds are tunable via `CLINIC_BARGEIN_*`
  env vars. **Known limitation:** the local mic has no acoustic echo cancellation, so sustained
  bot echo can still self-trigger; robust barge-in needs an AEC transport (WebRTC/telephony).
  End-to-end barge-in is audio-timing behavior and is verified on a live mic, not in the headless
  eval; the gate's decision logic has unit tests (`agent/tests/test_barge_in.py`).

**Phase 5 — telephony (done)** (LiveKit SIP, `+14842950169`): `MODE` switch in
`config.py`/`pipeline.py` (default `local`), LiveKit transport + join-token in `pipeline.py`,
telephony greeting with consent + explicit PII-minimization rule in `prompts.py`, and the
idempotent `agent/scripts/setup_livekit_sip.py` (Direct dispatch → room `clinic-inbound`).
Verified by live inbound calls that connected, ran the full booking flow, and returned a
confirmation number.

**Phase 7 — README + demo (done).** `README.md` rewritten as the recruiter-facing entry point
(Try-it phone number up top, ASCII architecture diagram, tech-stack + latency + 19/19 eval tables,
five "production-shaped" trust factors, resume line); `docs/demo_script.md` added (three-scenario
recording guide + QuickTime tips); final hygiene pass (stale scaffolding TODOs in `config.py`/
`prompts.py` resolved, gitignore/secret checks confirmed). The demo recording link is the one
remaining manual step (the author records the call and drops the URL into the README Try-it
section — placeholder is in place).
