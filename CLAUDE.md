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
| Telephony (later) | Inbound PSTN/SIP call ingress | Twilio / LiveKit SIP |
| Transport | Real-time audio in/out | Pipecat transport (WebRTC local dev → SIP later) |
| ASR | Speech → text | Deepgram (`pipecat.services.deepgram`) |
| LLM | Reasoning / dialogue | Anthropic Claude Haiku 4.5 (`pipecat.services.anthropic`) — see LLM-provider note below |
| TTS | Text → speech | Cartesia (`pipecat.services.cartesia`); Piper for free-tier testing (TODO) |
| Orchestration | Pipeline, turn-taking, context | Pipecat (`Pipeline` / `PipelineWorker` / `WorkerRunner`) |
| Scheduling backend | Availability / hold / booking | FastAPI mock service (`scheduling_api/`) |
| Storage | Slots & bookings | SQLite (`scheduling_api/clinic.db`) |

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
│   ├── .env.example           # API-key placeholders
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

## 8-phase build plan

- **Phase 0 — Scaffold, docs, mock scheduling API.** ← *(current)*
- **Phase 1 — Live ASR→LLM→TTS loop.** Local/WebRTC transport, core dialogue happy path.
- **Phase 2 — Tool/function calling.** Wire the agent to the scheduling API
  (availability / hold / confirm).
- **Phase 3 — Dialogue management hardening.** State machine (Pipecat Flows), input
  validation, per-state fallback/no-match handling.
- **Phase 4 — Turn-taking / barge-in.** VAD tuning, interruption handling.
- **Phase 5 — Observability.** Per-turn latency, ASR confidence, tool-success metrics,
  structured logs.
- **Phase 6 — Eval harness.** Scripted + adversarial test conversations.
- **Phase 7 — Telephony + governance + deploy.** Twilio/LiveKit SIP, AI disclosure &
  call-recording consent, deployment.

## Conventions & governance

- **Synthetic data only.** No real PHI in code, seeds, logs, or prompts.
- **Secrets** live in each service's `.env` (copied from `.env.example`), never committed.
- **AI disclosure** is mandatory in the greeting; call-recording consent is required once
  telephony lands (Phase 7). See `agent/src/clinic_agent/prompts.py`.
- Keep the agent and scheduling API decoupled — the agent talks to the API over HTTP.

## Running the services

```bash
# Mock scheduling API (works today)
cd scheduling_api && uv sync && uv run uvicorn app.main:app --reload
cd scheduling_api && uv run pytest

# Voice agent (Phase 1+; skeleton only in Phase 0)
cd agent && cp .env.example .env   # then fill in keys; uv sync when Phase 1 starts
```

## Current status

**Phase 4 complete.** Phases 0–3 (scaffold + mock API, live ASR→LLM→TTS loop, scheduling-API
tool calls, dialogue hardening + headless eval) are done. Phase 4 fixed all five confirmed
defects from the Phase-4 target list in [`docs/build_spec.md`](docs/build_spec.md) and added
barge-in:

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

**Next: Phase 5 — telephony** (Twilio/LiveKit SIP + governance/deploy). Not started.
