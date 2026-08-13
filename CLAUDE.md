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
| Orchestration | Pipeline, turn-taking, context | Pipecat (`Pipeline` / `PipelineWorker` / `WorkerRunner`) |
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

## Conventions & governance

- **Synthetic data only.** No real PHI in code, seeds, logs, or prompts.
- **Secrets** live in each service's local `.env` (git-ignored), never committed. Real API
  keys must never enter tracked files.
- **AI disclosure** is mandatory in the greeting; call-recording consent is required once
  telephony lands (Phase 7). See `agent/src/clinic_agent/prompts.py`.
- Keep the agent and scheduling API decoupled — the agent talks to the API over HTTP.
- **One container per voice session (concurrency).** A running agent process holds one live
  audio pipeline on the asyncio event loop; Python's GIL means multiple concurrent voice
  sessions must **not** share a single process — CPU-bound audio/VAD/serialization work on one
  call would stall the others. For now (single demo call, `MODE=telephony` with a Direct
  dispatch rule → one shared room) this is a non-issue. But Phase 6 Dockerization must run
  **one container per session**: switch the LiveKit dispatch rule from Direct to Individual
  (room-per-call) and start one agent process/container per room. Noted here so Phase 6 gets
  the isolation model right instead of trying to multiplex sessions in one process.

## Running the services

```bash
# Scheduling API — needs Postgres (Phase 9). Any throwaway database will do:
#   initdb -D /tmp/pgclinic -U postgres --auth=trust
#   pg_ctl -D /tmp/pgclinic -o "-p 55432 -c listen_addresses=127.0.0.1 \
#       -c unix_socket_directories=''" -l /tmp/pgclinic.log start
#   createdb -h 127.0.0.1 -p 55432 -U postgres clinic_dev   # and clinic_test, clinic_eval
export CLINIC_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/clinic_dev
cd scheduling_api && uv sync --extra dev && uv run uvicorn app.main:app --reload
cd scheduling_api && uv run pytest        # 35 tests; SKIPPED if no Postgres is reachable

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

# Voice agent — local mic/speaker (default, MODE=local)
cd agent && uv run python -m clinic_agent.pipeline

# Voice agent — telephony (LiveKit SIP, inbound calls to +14842950169)
cd agent && uv run python scripts/setup_livekit_sip.py   # one-time, idempotent trunk+rule
cd agent && MODE=telephony uv run python -m clinic_agent.pipeline
```

`MODE` (default `local`) is the only switch between the laptop mic/speaker path and the LiveKit
SIP telephony path — the ASR→LLM→TTS loop, tool calls, mic gate, and barge-in are identical in
both. Full telephony setup steps: `docs/build_spec.md` → *Phase 5 — Telephony (LiveKit SIP)*.

## Current status

**Project complete — all 7 phases (0–7) done.** Phases 0–5 are done (scaffold + mock API, live
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
