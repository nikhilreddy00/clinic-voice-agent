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
| LLM | Reasoning / dialogue | Groq-hosted Llama (`pipecat.services.groq`) |
| TTS | Text → speech | Cartesia (`pipecat.services.cartesia`); Piper for free-tier testing (TODO) |
| Orchestration | Pipeline, turn-taking, context | Pipecat (`Pipeline` / `PipelineWorker` / `WorkerRunner`) |
| Scheduling backend | Availability / hold / booking | FastAPI mock service (`scheduling_api/`) |
| Storage | Slots & bookings | SQLite (`scheduling_api/clinic.db`) |

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

**Phase 0 complete.** Scaffold, docs, and the mock scheduling API are in place and tested.
Phase 1 (the live ASR→LLM→TTS loop) has **not** started — `agent/` is a skeleton with TODO
markers only. Do not add pipeline logic until Phase 1 is explicitly begun.
