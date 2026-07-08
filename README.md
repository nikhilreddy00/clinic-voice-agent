# Clinic Voice Agent

**Production inbound voice scheduling agent — healthcare clinic themed, live on a real phone number.**

**Deployed on:** Railway (scheduling API + voice agent, always-on)

A caller phones the clinic, hears an AI disclosure, and books an appointment end to end — intake,
availability lookup, slot confirmation, and booking — over a real telephone number. Built as a
portfolio project to show the *shape* of a real voice system: dialogue design, tool calls,
observability, evals, and governance.

> **Synthetic data only.** No real patient data (PHI) enters this repo, its logs, or its prompts.

---

## 📞 Try it

**Call `+1 (484) 295-0169`.**

The agent runs 24/7 on Railway — no local setup needed to test it. Just call.

You'll be greeted by an automated AI assistant that discloses it's an AI and (on the phone path)
states a call-recording consent line. Then just talk to it like a receptionist:

- *"I'd like to book an appointment."*
- Give a name, a date of birth, whether you're a new or existing patient, and a reason
  (e.g. *"annual checkup"* or *"sore throat"*).
- It offers you open slots — pick one (*"Tuesday at 9 works"*).
- It reads the appointment back and asks to confirm. Say *"yes, book it."*
- You get a confirmation number and the call closes.

Deployment topology and redeploy steps are in [`docs/build_spec.md`](docs/build_spec.md) →
*Deployment (Railway)* — the agent is an outbound-only worker to LiveKit Cloud, so it needs no
inbound tunnel.

Note: the mock scheduling API always keeps ~3 upcoming working days of slots available (the seeded
window rolls forward automatically), so if asked for dates far in the future the agent will offer
the soonest available instead.

Expect a natural, interruptible conversation. Total call time is typically under 90 seconds. The
LLM is the main source of response latency at this stack (~3 s per turn — see
[Latency](#latency-real-numbers) below).

---

## Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │                Pipecat pipeline                │
                                  │        (one container per voice session)       │
   ☎  Caller                     │                                                │
   (PSTN phone)                   │   Deepgram        Claude Haiku 4.5   Cartesia  │
       │                          │     ASR      ──▶      LLM        ──▶    TTS     │
       │ audio                    │  (speech→text)   (reasoning/dialogue) (text→   │
       ▼                          │       ▲               │   │           speech)  │
┌───────────────┐   RTP/SIP  ┌────┴───────┴───────────────┼───┼──────────────┐    │
│  LiveKit SIP  │◀──────────▶│              turn-taking / barge-in            │    │
│  +14842950169 │  audio     │   VAD · mic gate · interruption · metrics taps │    │
└───────────────┘            └────────────────────────────┼───┼──────────────┘    │
                                  │                         │   │                   │
                                  └─────────────────────────┼───┼───────────────────┘
                                                            │   │ tool calls (HTTP)
                                                            ▼   │
                                             ┌──────────────────────────────┐
                                             │   Mock scheduling API (FastAPI)│
                                             │   GET  /availability           │
                                             │   POST /hold-slot              │
                                             │   POST /confirm-booking        │
                                             │   GET  /metrics · /dashboard   │
                                             │   SQLite (slots + bookings)    │
                                             └──────────────────────────────┘

  Observability: every turn → logs/calls.jsonl (ASR/LLM/TTS/E2E latency, ASR confidence,
  tool outcomes) → aggregated live by GET /metrics → rendered at /dashboard.
```

Audio flows in from the caller over PSTN → LiveKit SIP → the Pipecat pipeline, through
Deepgram (ASR) → Claude Haiku (LLM) → Cartesia (TTS), and back out the same path. Whenever the
dialogue needs real data — open slots, a hold, a booking — the LLM makes a tool call to the mock
scheduling API over HTTP. `MODE=local` swaps LiveKit SIP for the laptop mic/speaker; everything
downstream of the transport is identical.

## Tech stack

| Layer | Responsibility | Tech |
|-------|----------------|------|
| Telephony | Inbound PSTN/SIP call ingress, real phone number | **LiveKit SIP** (free tier, `+14842950169`) |
| Orchestration | Pipeline, turn-taking, context, barge-in | **Pipecat** |
| ASR | Speech → text (+ confidence) | **Deepgram** |
| LLM | Reasoning / dialogue / tool calls | **Anthropic Claude Haiku 4.5** |
| TTS | Text → speech | **Cartesia** |
| Scheduling API | Availability / hold / booking | **FastAPI + SQLite** (mock backend) |
| Eval | Headless conversation scoring | Scripted harness, **structural tool-trace scoring** |
| Observability | Per-turn latency + outcomes + dashboard | JSONL sink + `GET /metrics` + `/dashboard` |

## Latency (real numbers)

Captured from a live inbound phone call (Phase 6), voice-to-voice, per turn:

| Stage | P50 | P95 |
|-------|-----|-----|
| **ASR** (Deepgram) | 87 ms | 171 ms |
| **LLM** (Claude Haiku 4.5) | 2925 ms | 4060 ms |
| **TTS** (Cartesia) | 134 ms | 1078 ms |
| **End-to-end** (voice → voice) | 4098 ms | 5247 ms |

**Honest framing:** ASR and TTS are excellent — both sit comfortably in the low-hundreds-of-ms
range and are not the bottleneck. **The LLM dominates end-to-end latency** (~3 s median, the bulk
of the ~4 s E2E). This is a property of the chosen stack, not the pipeline: Claude Haiku 4.5 was
selected during development to unblock eval iteration (Groq's free tier caps at 100K tokens/day,
which a full eval run exhausts). For a latency-optimized production build, **swapping the dialogue
LLM to Groq-hosted Llama would cut LLM latency to roughly 200–400 ms** and bring E2E under ~500 ms.
The swap is a documented, one-edit change (`agent/src/clinic_agent/pipeline.py` + `eval/run_eval.py`);
there is no runtime provider flag by design, to keep the active path simple.

## Eval results

**19/19 cases pass — 100% task completion, 100% overall pass** (0 errored), on Claude Haiku 4.5.

The suite covers happy paths, edge cases, and **adversarial** conversations (mumbled/vague input,
mid-sentence changes of mind, pure nonsense then recovery, unsupported intents, off-topic medical
detail, human-handoff requests, and symptom-minimization probes). Scoring is **structural — it
inspects the tool-call trace** (did the agent call `check_availability` / `hold_slot` /
`confirm_booking` in the right shape, with the right arguments, and reach the expected outcome?)
rather than matching transcripts. That makes the eval robust to benign phrasing differences while
still catching real defects like fabricated confirmations or wrong calendar dates.

Phase-3 baseline was 88% task completion / 81% overall pass; Phase-4 dialogue hardening fixed all
five confirmed defects (plus a fabricated-booking bug) and took it to 100% / 100%.

## What "production-shaped" means here

Five trust factors that separate a demo from something you'd let answer a real phone:

1. **Latency percentiles** — per-turn ASR/LLM/TTS/E2E timings with P50/P95/P99, so performance is
   measured, not guessed, and the bottleneck is named honestly.
2. **Barge-in with a false-positive metric** — the caller can interrupt the agent mid-sentence;
   a sustained-speech mic gate plus a minimum-words turn strategy suppress self-triggering, and
   false-positive interruptions are counted rather than hand-waved.
3. **Eval harness** — 19 scripted + adversarial conversations scored on the tool-call trace, run
   headlessly so regressions surface before a live call does.
4. **Observability dashboard** — a structured JSONL sink feeds `GET /metrics` and a live
   `/dashboard` showing latency percentiles, ASR confidence, tool success, outcomes, and recent
   calls, refreshing every 5 s.
5. **Governance** — mandatory AI disclosure, call-recording consent on the telephony path,
   PHI-minimization rules in the prompt and in logs, and a human-escalation path for anything
   out of scope.

## Features

- **AI disclosure + call-recording consent** — spoken deterministically in the greeting, never
  LLM-generated; consent line added on the telephony path.
- **Richer clinical intake** — collects date of birth, new-vs-existing patient status, and a
  coarse symptom/visit reason, without soliciting detailed clinical narrative.
- **PHI minimization in logs** — the JSONL sink redacts sensitive fields; the prompt forbids
  reading a name and phone number back together or repeating DOB/IDs unprompted.
- **Human escalation path** — unsupported intents (billing, prescriptions, clinical questions) and
  stuck callers are handed off rather than force-fit into a booking.
- **Barge-in with false-positive tracking** — Pipecat interruption gated by a mic gate + minimum-
  words strategy, with a false-positive-interruption metric.
- **Per-turn latency logging** — voice-to-voice timing for every turn, summarized to P50/P95/P99
  at hangup.
- **One-container-per-session design** — the concurrency model is explicit: one agent process per
  live audio session (documented for the Individual-dispatch, room-per-call scaling path).

## Local setup

The agent is already deployed and live — local setup is only needed for development.

Two independent services, each managed with [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Mock scheduling API (FastAPI + SQLite) — serves :8000, /docs, /metrics, /dashboard
cd scheduling_api && uv sync && uv run uvicorn app.main:app --reload
cd scheduling_api && uv run pytest          # run the test suite

# 2. Voice agent — local mic/speaker (default, MODE=local)
#    create agent/.env with your Deepgram / Anthropic / Cartesia / LiveKit keys first
cd agent && uv sync && uv run python -m clinic_agent.pipeline

# 3. Voice agent — telephony (LiveKit SIP, inbound calls to +14842950169)
cd agent && uv run python scripts/setup_livekit_sip.py    # one-time, idempotent
cd agent && MODE=telephony uv run python -m clinic_agent.pipeline
```

`MODE` (default `local`) is the only switch between the laptop mic/speaker path and the LiveKit
SIP telephony path — the ASR→LLM→TTS loop, tool calls, mic gate, and barge-in are identical in
both.

**Always-on demo (deployed):** both services run on **Railway**, so `+14842950169` is answered
24/7 with nothing running locally — the agent is an outbound-only worker connecting to LiveKit
Cloud, and the scheduling API is a public Railway URL. Redeploy steps and topology are in
[`docs/build_spec.md`](docs/build_spec.md) → *Deployment (Railway)*.

**One-command local demo:** `./start_demo.sh` starts the API + the agent in `MODE=telephony` and
prints the dashboard URL; run `ngrok http 8000` in another terminal to expose the dashboard
publicly. Use this when iterating locally.

Full telephony provisioning and the demo sequence are in
[`docs/build_spec.md`](docs/build_spec.md) (Phase 5, Phase 6 & Deployment). Docker is also provided
(`docker compose up --build`) — see the Known limitations note.

## Known limitations

- **Seed slot window.** The mock backend keeps ~3 upcoming working days of slots, rolled forward
  automatically on each `/availability` read so an always-on deploy never runs out (see
  `db.refresh_available_slots`). Seed slot *hours* are stored as UTC while the agent reasons in
  clinic-local (`America/New_York`) time, so a mid-day caller can see an early-morning slot filtered
  out that still reads as upcoming locally. Requests far outside the window ("sometime next month")
  have nothing to offer and fall back to the soonest available slot.
- **Docker not locally tested.** The Dockerfiles and `docker-compose.yml` are present and validated
  (compose config parses, standard uv build) but not built locally (no Docker on the dev machine).
  The live demo runs on the host via `start_demo.sh`, which *is* fully tested.
- **Rescheduling is v1.1.** The mock API has no patient-lookup endpoint, so a returning caller
  re-collects intake and books a fresh slot rather than amending an existing appointment.
- **LLM latency vs Groq.** Median E2E is ~4 s, LLM-bottlenecked (see [Latency](#latency-real-numbers)).
  A Groq-hosted Llama swap is documented as the path to a sub-500 ms latency-optimized build.
