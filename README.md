# Clinic Voice Agent

A production-shaped inbound appointment-scheduling voice agent for a healthcare clinic
(portfolio project). A caller phones in; the agent greets them with an AI disclosure,
collects intent and details, offers open appointment slots, confirms, books, and closes.

> **Synthetic data only.** No real patient data (PHI) belongs in this repo, its logs, or its
> prompts.

See [`CLAUDE.md`](CLAUDE.md) for architecture and the phase plan, and
[`docs/build_spec.md`](docs/build_spec.md) for the dialogue state machine.

## Status

**Phase 0 (scaffold) complete.** The mock scheduling API works and is tested. The voice
agent (`agent/`) is a skeleton with TODO markers — no ASR/LLM/TTS logic yet.

## Services

This repo contains two independent services, each with its own `pyproject.toml` (managed
with [`uv`](https://docs.astral.sh/uv/)).

### `scheduling_api/` — mock scheduling backend (working)

FastAPI + SQLite. Endpoints: `GET /availability`, `POST /hold-slot`, `POST /confirm-booking`.

```bash
cd scheduling_api
uv sync
uv run pytest                                  # run the test suite
uv run uvicorn app.main:app --reload           # serve on http://127.0.0.1:8000
```

Try it (availability → hold → confirm):

```bash
# 1. List open slots
curl -s http://127.0.0.1:8000/availability | python3 -m json.tool

# 2. Hold a slot (use a slot_id from step 1)
curl -s -X POST http://127.0.0.1:8000/hold-slot \
  -H 'Content-Type: application/json' \
  -d '{"slot_id": 1}' | python3 -m json.tool

# 3. Confirm the booking (use the hold_id from step 2)
curl -s -X POST http://127.0.0.1:8000/confirm-booking \
  -H 'Content-Type: application/json' \
  -d '{"hold_id": "<hold_id>", "patient_name": "Jane Doe", "reason": "annual checkup"}' \
  | python3 -m json.tool
```

Interactive docs at http://127.0.0.1:8000/docs once the server is running.

### `agent/` — Pipecat voice agent (skeleton, Phase 1+)

```bash
cd agent
cp .env.example .env    # fill in Deepgram / Groq / Cartesia / LiveKit keys
# uv sync + implementation land in Phase 1
```
