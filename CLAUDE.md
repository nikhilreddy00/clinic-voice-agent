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
| Telephony (later) | Inbound PSTN/SIP call ingress | LiveKit SIP (free tier, one number included: `+14842951203`) |
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
  `+14842951203`), `MODE=local|telephony` runtime switch, idempotent SIP trunk + Direct-dispatch
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
  done. **First real runs executed 2026-09-03** (partial: 3 of 19 cases, Haiku + Groq).
  - **Prompt caching NOW ENGAGES on Haiku 4.5 — this reverses the Phase-8 finding.** The
    cacheable prefix was 3,811 tokens against the 4,096 minimum; Phase 13's six new tools pushed
    it to **5,023**, so the breakpoint is live. Measured over 3 cases: TTFT p50 830 → 642 ms,
    p95 1,927 → 988 ms, **$0.0705 → $0.0195 a call**, 145,700 tokens served from cache.
    `CLINIC_PROMPT_CACHE` now defaults to ON (`=0` opts out).
  - Caching is **per-intent**, and only two intents clear the floor: `None` (turn one) and
    `schedule_appointment`, both 5,023. Reschedule is 3,366, cancel 2,778, refill 2,293,
    everything else under 1,700. So it pays on new bookings and is inert elsewhere — a zero
    cache-read on a refill call is expected, not a bug. Full table in `core/adapters/llm.py`.
  - **Still do NOT shrink the booking prompt.** There are only ~900 tokens of headroom over the
    floor; trimming it turns the optimization off silently, with no error.
  - **Groq is NOT the dialogue LLM, and the measurement says don't make it one.** On the real
    5,023-token booking prompt: cached Haiku TTFT p50 **621 ms** vs Groq Qwen3.8 **4,614 ms**
    and gpt-oss-120b **4,647 ms** (free tier, 429s included; best-case single samples were
    745/559 ms — still no better). Groq has no prompt caching, so it pays full prefill every
    turn while Haiku reads 5k tokens back at 0.1x. **Groq's speed advantage is real on SHORT
    prompts and disappears on long cached ones** — which is why it wins for the classifier
    (below) and loses here.
  - **Groq's Llama fleet is dead.** `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were
    deprecated 2026-06-17 and shut down 2026-08-16 — confirmed against the live `/models`
    endpoint. Both `config.py` and `eval/providers.py` pointed at the first one. The served
    fleet is `openai/gpt-oss-{20b,120b}` and `qwen/qwen3.{6,8}-27b`. **`agent/.env` still has a
    stale `GROQ_MODEL=llama-3.3-70b-versatile` — change it or unset it.**
  - `eval/providers.py` set no SDK timeout or retry cap, so a rate-limited candidate hung a
    ONE-CASE run for six minutes with zero output. Now `BAKEOFF_TIMEOUT_SECS` (60) and
    `BAKEOFF_MAX_RETRIES` (1).
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
    there is no code path that could do otherwise. **The failure mode is the opposite one and it
    has happened: the agent SAYING it filed a refill without calling the tool at all** — see the
    2026-09-04 fabrication note below. "Never approves" is enforced by the API; "never claims to
    have filed" is enforced by `_ACTION_CLAIM`.
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

**RESTART THE SCHEDULING API AFTER CHANGING IT.** Python imports a module once; editing
`app/db.py` does nothing to a process that is already running, and `uvicorn` is started here
WITHOUT `--reload`. This cost a full round of live calls: the shared-phone fix was written at
17:19, the API serving the calls had been up since 12:57, and the caller hit the original
unfixed bug at 17:52 — identical trace, no new defect, the fix simply was not loaded.
`migrate_patient_identity` runs at boot, so an un-restarted API also means an un-migrated
database. After ANY change under `scheduling_api/`, restart it and re-check `/health`.

**The follow-through nudge must not fire on a turn whose tool already ran (2026-09-03).** The
dead-air the caller heard as "the bot stopped speaking, so I said hello".

`reducer._ACTION_CLAIM` re-prompts a turn that announces an action and calls no tool. It tested
`committed` — the tool calls made by THIS REQUEST — but `turn_tool_uses` is cleared at the end
of every request and `committed_tool_ids` is cleared when results are flushed back, so after a
successful tool round trip the follow-up request that narrates the result looked like an empty
promise. Trace 20260903T215242896036Z at t=130.4: the caller said "Yeah. Sure.", `hold_slot`
succeeded, and the read-back "I'm holding that for you. So I have you as Olivia. Your date of
birth is…" triggered a nudge. Starting that extra request closed the live TTS context —
Cartesia reported `Context closed`, `BotStoppedSpeaking` arrived with `completed: false` **1.06
seconds into an ~8 second sentence**, and the nudge's own reply was queued to the dying context
and never spoken at all. The caller heard "…So I have you as Ol—" then 6.5 s of silence, on the
confirmation read-back, and said "Hello?" to get the agent back.

  - `CallState.turn_had_tool` is set on any tool use (invoked OR gate-refused) and reset with
    `nudged` on each new caller turn. The rule was always per CALLER TURN; only the mechanism
    was per request.
  - **And a reply that ENDS IN A QUESTION is never nudged (2026-09-04).** Same truncation, third
    variant, found in trace `20260904T173146047919Z` at t=34.0. A returning caller opened with
    "I need to reschedule"; the model said *"Good to hear from you — I'm happy to help move
    that. Let me pull up your appointment first. What's your full name?"* and called no tool
    **because it could not** — every PHI tool is gated behind `verify_identity`. So it did the
    only correct thing available, which is verbatim what `_ACTION_NUDGE` demands: it asked for
    what it needed. The nudge fired anyway, closed the live TTS context, and cut the sentence
    off 0.8 s into ~6 s. The caller heard *"Good to hear from you — I'm happy to h—"* and said
    "Hello?". **A reply that asks the caller a question is not silence.** The comment claiming a
    false positive costs "one extra request" was the wrong cost model and is corrected in place;
    that wrong model is why this shipped twice.
  - **A `ProviderDegraded` tts `Context closed` with no preceding `SpeechStarted` is the
    signature.** With a barge-in it is normal (CLAUDE.md, Phase-14 notes); without one it means
    the engine cut the agent off mid-sentence and the remaining text was dropped.

**`scripts/reset_demo_data.py` — fresh slots, no appointments.** Dry-run by default, `--yes` to
apply. Slots are DELETED and regenerated rather than re-seeded: `db._seed` inserts them with
fixed ids and `ON CONFLICT (id) DO NOTHING`, which is right for boot-time idempotency and means
re-running it leaves every stale `start_time` untouched. Keeps `clinics`/`providers`/
`clinic_facts` — those are the clinic, not call data.

**Testing without a phone — `./run_e2e.sh`.** Live calls cost Cartesia/Deepgram/LiveKit credit
and the author has one test handset, so **every identity scenario must be provable offline**.
`agent/tests/test_shared_phone_e2e.py` drives a scripted call through the REAL reducer, the REAL
`ToolExecutor`, real HTTP, and real Postgres — only the model and the microphone are scripted.

This layer exists because the previous suites could not have caught the last three defects.
`scheduling_api/tests` covers HTTP-to-Postgres; `test_reducer` covers the reducer as a pure
function with fake tool results. Neither sees the SEAM between them — argument injection, the
identity gate, the order tools fire in — and the seam is where every live failure has been.

  - **`./run_e2e.sh` is the gate before placing a call.** It brings up a throwaway database and
    an API on its own port, runs all four suites plus the emergency detector, and tears down.
  - **The e2e suite REFUSES to run unless the API reports the scratch database by name.** It
    books and truncates, and the default port is the one a developer already has pointed at
    real data. Wrong database is a skip, never a run.
  - The reset fixture is not hygiene: without it every test inherits the previous one's bookings
    on the SAME number, so "Nick can see his appointments" passes while showing six of them —
    exactly the shape of leakage these tests exist to detect.

**Identity is per-person, not per-call (2026-09-03).** `patient_name` was promoted from the API
result while `verified_dob` was stashed only while UNVERIFIED — an anti-tampering rule whose
mechanism let the two fields describe DIFFERENT PEOPLE. Verify as Joe, then verify as Nick, and
the engine reported the caller as Nick (the prompt literally says "their name is Nick") while
every PHI call still carried Joe's date of birth and read Joe's chart. The agent would have said
"Nick, your appointment is…" and read out Joe's appointment. On a single shared handset this is
the likeliest cross-person disclosure in the system.

  - `CallState.submitted_dobs` tags each submitted date with its `tool_call_id`, and
    `_on_tool_completed` promotes the name and the date **from the same call the API accepted**.
    Both together or neither. A FAILED attempt still promotes nothing, which was the whole point
    of the original rule.
  - A caller on a shared handset may legitimately verify as a second person mid-call. When they
    do, every later PHI call must follow them.

**`caller_memory` counts the household, not one member (2026-09-03).** It did `GROUP BY p.id`
and took the first row, so a number with three confirmed appointments reported two — and which
two depended on row order. The number is the right unit (this runs BEFORE verification; nobody
has said who they are yet) and it still returns nothing identifying.

**Known limitation, deliberately not fixed: two people sharing a phone AND a birthday.** Twins
land on one row — the later booking renames it, and verifying as either returns that name and
both sets of appointments. Adding `name` to the key would fix twins and break something
commoner: "Nick" on one call and "Nicholas Kumar" on the next is one person, and a name-keyed
record files them as two. Spoken names are not stable enough to key on; birthdays are. Pinned by
`test_two_people_sharing_a_phone_AND_a_birthday_collapse_into_one_record` so it stays a decision.
The real fix, if it is ever needed, is a patient identifier the caller states — not a fuzzier
name match, which weakens a credential to solve a data-modelling problem.

**A phone number is a household, not a person (2026-09-03).** The highest-severity defect
found so far, and it was a data-model bug rather than a prompt or engine one.

`patients` was `UNIQUE (clinic_id, phone)` — one identity per number, forever. Two rules that
are each individually correct combined into a permanent lockout AND a PHI mix-up:

  * `_upsert_patient` refused to overwrite a `date_of_birth` already on file (right: the DOB is
    the verification secret, and a later booking that mistyped it would lock the real patient
    out), but it did NOT protect `name`; and
  * `_verify` stops dead when an enrolled number presents a wrong DOB (right: falling through
    to the name+DOB search would let anyone holding an enrolled handset reach a stranger's
    chart by naming them).

So the first caller from a number owned it. **Live:** "Joe" (DOB 03/05/2001) booked from a
number already enrolled to "Nick" (DOB 05/08/2003) and was given confirmation E938C8F6. The row
became a MERGE — renamed *Joe*, still carrying *Nick's* DOB, owning *Nick's* bookings. Joe
called back and was refused twice on the exact date of birth he had just booked with. The
lockout is the visible half; the disclosure is the worse half — verifying with Nick's DOB would
have returned Joe's name and Nick's appointments.

  - **The key is now `UNIQUE (clinic_id, phone, date_of_birth)`.** A different person on the
    same handset is simply a different row, so no `COALESCE` guard is needed — nothing can
    reach another person's record to overwrite it.
  - **`patients.date_of_birth` now stores the NORMALIZED form** (`normalize_dob`). The column
    used to hold whatever the caller said, normalized on every comparison; that is fine for
    comparing and useless for a key, where "3/5/2001" and "03/05/2001" file one person twice.
  - **`_verify` matches the DOB against EVERY patient on the number**, then keeps the original
    hard stop when none match. Widening who can be found must not widen who gets in — there is
    a test for exactly that (`..._cannot_be_used_to_reach_a_stranger_by_name`).
  - **`db.migrate_patient_identity` runs on every boot** and is idempotent. It has to be Python,
    not SQL in `schema.sql`: the stored dates must be canonicalized BEFORE the new key exists,
    and canonicalizing can itself create duplicates that must be merged first.
  - **`scripts/repair_patient_identities.py` un-merges rows that already exist** — dry-run by
    default. The migration cannot do this: nothing in `patients` records who the second person
    was, but `bookings` carries its own `patient_name`/`date_of_birth`, so a booking that
    disagrees with its patient row belongs to someone else. Verified against a rebuilt copy of
    the live corruption: both callers verify afterwards and each sees only their own bookings.
  - 8 new tests in `tests/test_verified_flows.py` covering shared handsets end to end
    (verify / list / reschedule / cancel / third-DOB refusal / stranger-by-name refusal /
    same-birthday-spoken-differently). **4 of them fail against the old lookup.**

**Live-call hardening round 2 (2026-09-03).** Two calls placed — a booking and a "reschedule"
— both of which *sounded* perfect. Reading `logs/traces/` found two defects neither the audio
nor the console log showed. Do not regress these:

  - **`intents.SCHEDULING_PREEMPT_BLOCKED` — `schedule_appointment` never preempts an
    established scheduling flow.** This is the double-book fix, and it is the QUESTION_INTENTS
    lesson one level in. A reschedule and a cancellation both CONTAIN picking a time, so
    slot-fill answers inside them ("Tuesday.", "Yeah, sure.") are word-for-word what starting a
    new booking sounds like, and the classifier — which sees one utterance with no history —
    cannot tell them apart. Live: an intent established at `reschedule_appointment` flipped to
    `schedule_appointment` at 0.85 (exactly `INTENT_SWITCH_CONFIDENCE`), which swapped the tool
    set from `{list, check_availability, reschedule}` to `{check_availability, hold_slot,
    confirm_booking}`. The model instantly lost the ability to reschedule and gained the ability
    to book, so it booked. The caller heard a reschedule confirmation and hung up with **two
    live appointments** (104BF84D Sept 7, 38510718 Sept 8). Raising the threshold cannot fix
    this — the bar was already 0.85 and the classifier was at 0.85. **The block is
    one-directional on purpose:** `cancel` and `reschedule` still preempt, because those are
    distinctive phrases and neither can create an appointment nobody asked for.
  - **`CallState.active_hold_id` — the reducer owns the hold_id, the model never retypes it.**
    A hold_id is a 36-character random UUID with no redundancy; every character is load-bearing
    and none of it can be inferred. Live: `hold_slot` issued
    `c1154b66-3d70-4585-908c-2aa92646049f` and the model sent `...2aa92642049f` to
    `confirm_booking` — one hex digit, 6 → 2. The API correctly 409'd, the caller heard a
    stumble, and the whole hold-and-confirm round trip ran again: **30 seconds of the call spent
    re-doing work that had already succeeded.** Injected and overwritten in `_scoped_arguments`
    exactly like `phone` and `date_of_birth`. Spent on a successful confirm, kept on a failed
    one (so a retry reuses the hold rather than re-holding), and left alone when there is no
    live hold so the API's own error still speaks. **This was NOT a TTL problem** — the TTL is
    120 s and the gap was 14 s; do not "fix" it by raising the TTL.
  - **`CLINIC_FAST_BASE_URL` switches the classifier to the OpenAI wire format** (Groq /
    Cerebras) without touching the prompt, the enum, the forced tool call, or the events. This
    is where Groq actually wins: on the classifier's short prompt, `qwen/qwen3.8-27b` measured
    **p50 203 ms / p95 399 ms, 13/13 clean** against Haiku's **888 / 1,268**. Two caveats that
    are the reason it is **not** the default: `openai/gpt-oss-*` are REASONING models that spend
    `max_tokens` thinking before the forced tool call and get truncated mid-JSON at the
    classifier's 128-token ceiling (400 `tool_use_failed`) — qwen does not; and **Groq's free
    tier rate-limited this at ~12 concurrent requests and burned its daily budget in two
    benchmark runs.** A 429 mid-call is survivable (the reducer falls back to the unscoped
    prompt) but means no intent scoping at all. Worth switching on a paid tier, not before.
  - **`scripts/inspect_call.py` now prints the per-stage latency table** — endpointing,
    dispatch, LLM TTFT, speech queue, E2E, plus the parallel classifier. Use it after every
    call. A single voice-to-voice number cannot say whether a slow call is the model, the
    endpointer, or the speech queue, and those have completely different fixes.

**Measured on the 2026-09-03 calls** (`inspect_call.py`, 23 turns across two calls):

| stage | p50 | p95 |
|---|---|---|
| endpointing (speech stop → transcript) | 372 ms | 475 ms |
| dispatch (transcript → LLM start) | 1 ms | 2 ms |
| **LLM TTFT** | **855 ms** | 3,699 ms |
| speech queue (first token → speaking) | 380 ms | 949 ms |
| **E2E voice-to-voice** | **1,642 / 1,973 ms** | 2,522 / 4,673 ms |
| classifier (parallel) | 911 ms | 1,036 ms |

LLM TTFT was **52% of every turn** — which is why prompt caching, not endpointing and not TTS,
was the thing worth changing. The endpointing grace is measured at 372 ms p50, NOT the ~800 ms
this file previously implied; it is not currently the bottleneck and should not be the next
target. **Not yet re-measured on a live call after the caching change** — that needs a phone.

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
cd scheduling_api && uv run pytest        # 90 tests; SKIPPED if no Postgres is reachable

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

cd agent && uv run pytest        # 389 tests (9 e2e SKIP without a scratch API)
./run_e2e.sh                     # everything, incl. the e2e seam. Run before any live call.

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
Phases 9 (Postgres + Supabase), 10 (in-house event loop), 11 (concurrency + load proof), 12
(reasoning layer), and 13 (memory + verified tool surface) are done. Phase 8's bake-off harness
has now had its **first real runs** (partial — 3 of 19 cases) and produced the prompt-caching
result below.

**Phase 13's exit criterion is MET — see *Phase 13 CLOSED* below.** It took six live attempts;
five were defeated by the defects recorded here, and the sixth by an un-restarted API serving
pre-fix code. Everything below the microphone was already proven offline by `./run_e2e.sh`
(real reducer, real tool executor, real HTTP, real Postgres) and 90 API tests — which is the
lesson: the offline suites were necessary and were never sufficient, because every defect lived
in the seam between components or in the audio timing above them.

**Before the next live call, in this order:** reset the data
(`scheduling_api/scripts/reset_demo_data.py --yes`), **restart the scheduling API** so the
migration runs, restart the agent, then dial. See the restart note under *Live-call hardening*.

**Phase 13 CLOSED (2026-09-04).** Two live calls: a booking, then a callback that was
recognised (`known=True upcoming=1`, no name spoken), verified, listed, and rescheduled —
`verify_identity → list_appointments → check_availability → reschedule_appointment →
get_clinic_info`, one row in Postgres, no double-book. Trace `20260904T173146047919Z`. Still
untested on a phone: `cancel_appointment`, `request_refill`, the identity gate refusing on the
wire, and the scripted emergency path (test that with `run_intent_eval.py --detector-only` —
free, and it never touches a model anyway).

**Phase 14 — latency finish (in progress).** Plan:
`/Users/uvnikhil/.claude/plans/async-napping-comet.md`. Measured 2026-09-04, 29 turns, **the
first numbers taken after prompt caching**: endpointing 402/497 ms, dispatch 1 ms, **LLM TTFT
627/730 ms** (was 855 — caching confirmed live), speech queue 268/340 ms, **E2E 1,556/1,625 ms**
(was 1,642/1,973). Four findings, three of which contradict the Phase-8 plan:

  - **The "speech queue" is the LLM, not TTS.** First token → first sentence-ending period is
    205–255 ms p50; Cartesia + playback is only ~60–85 ms. The agent is waiting for a period,
    not for audio. Sentence-boundary TTS chunking — the Phase-8 plan's item — **was already
    built** in Phase 10.
  - **The delta bursts are the wire, not our event loop. Measured, not assumed.** Deltas arrive
    as one character, a 100–350 ms gap, then 50–100 characters at once. A probe that streams the
    real 5,023-token booking prompt with the loop otherwise idle — no media, no VAD, no STT —
    reproduces the same gaps at **loop lag p50 1.08 ms, max 29.6 ms**. Haiku's streaming is
    coarse at the source (3–6 deltas for a whole reply). Consequences: there is no blocked loop
    to unblock; speaking earlier only helps when a clause lands in an EARLIER BURST than the
    period, which is why the same change is worth 115 ms on one call and 32 ms on the next; and
    speculative LLM start is not undermined by anything on our side. Re-measure with
    `CLINIC_LOOP_LAG=1` (`[loop] blocked N ms` lines) plus the streaming-shape table in
    `inspect_call.py` before believing otherwise.
  - **The FAST tier is dead in the dialogue path.** All 22 routed turns: `standard/haiku-4.5`.
    `select_tier` only picks FAST for `Intent.EMERGENCY`, which is scripted and never reaches a
    model. Left alone deliberately — a fast tier pays only once there is a fast model worth
    pointing it at, and Phase 8 says there isn't one.
  - **One Anthropic client per process, not per call** (`llm.shared_anthropic_client`). Each
    session used to build its own for the dialogue AND the classifier: 2N pools per worker and
    two TLS handshakes per call. Turn-1 TTFT was 736–740 ms against 540–670 steady. **An adapter
    now closes only a client it constructed itself** — closing an injected one would tear the
    pool out from under every other call in the worker, which is the whole risk of sharing it.

**`reducer._FIRST_CLAUSE_MIN_CHARS` — the agent starts speaking at the first CLAUSE of a reply,
not the first sentence.** Opening chunk only (`state.utterance_id is None`): mid-reply there is
already audio playing, so an early split buys nothing and costs prosody. **The 20-character
floor is an underrun guard, not a style rule** — the opening chunk must take longer to SPEAK
than the next burst takes to ARRIVE (~335 ms p95), or the caller hears a stutter mid-phrase
instead of a late start. It is a plain constant and **not an env var on purpose**: `reduce()` is
pure so traces replay identically anywhere, and reading the environment there would make one
trace produce different `Speak` actions on different machines.

**Endpointing is CLOSED — do not adopt semantic EOU (measured 2026-09-04, 14 calls / 126
turns).** It was the Phase-8 plan's headline Phase-14 item and the measurement kills it three
ways. **83% of turns never buy grace at all** and sit at 372 ms p50, which is Deepgram's 300 ms
window plus ASR finalization — a turn-detector reads the same audio, cannot remove that, and
*adds* ~20 ms. **The grace rule costs 5.5 s across every call ever recorded**, i.e. 47 ms
averaged over all turns, so there is nothing there to reclaim. And it is **already 90% precise**
— 19 of 21 windows were bought by a caller who then kept talking; the 2 misses were a trailing
comma and an ASR error, both ambiguous to a human. The `weak` 0.7 s tier was perfect (8 held, 0
wasted), which is the tier that exists so "December eight two thousand" is not taxed. Lowering
Deepgram's `endpointing=300` is the only lever left on that floor and it manufactures fragments
for the grace rule to catch — a trade, not a win. Full table: `docs/build_spec.md` → *Phase 14*.

**Phase 14's exit criterion is now E2E p50 ≤ 1,200 ms / p95 ≤ 2,000 ms (decided 2026-09-04).**
It replaces the inherited ≤ 800 ms, which was set in Phase 8 before anything was measured and is
not reachable on this stack: steady-state TTFT on an ALREADY-CACHED prompt is 540–670 ms, most of
an 800 ms budget before endpointing, TTS, or the network take a share. Phase 8 measured the
obvious escape as worse (Groq, no prompt caching: 4,614 ms on the same prompt). **Going below
~1,000 ms needs a different model host — in-region Bedrock/Vertex Haiku — not another
orchestration change.** Say that plainly rather than re-tuning the engine against a number the
provider decides. Best measured so far: **1,385 ms p50** (2026-09-04 refill call, 8 turns).

**A refill the agent ANNOUNCED and never filed (2026-09-04).** Trace
`20260904T183444349772Z` at t=123.5 — a live call that cancelled an appointment correctly and
then fabricated the next task. The classifier had switched to `medication_refill` at 0.95, so
`request_refill` **was** in the tool set (req-13 routed as "medication_refill dialogue"); the
model called nothing, `stop_reason: end_turn`, and said *"Got it — I'll send a refill request
for ibuprofen to our staff, and they'll review it and follow up with you."* No `staff_tasks`
row exists. The caller thanked it and hung up believing a refill was queued.

  - **`reducer._ACTION_CLAIM` is a verb list, and a verb list is always one live call behind the
    model's vocabulary.** It had `check|look|see|pull|find|book|hold` and not `send` — and
    *sending / filing / forwarding / passing along to staff* is the entire shape of the
    non-scheduling tools, so those were the most consequential verbs to be missing. Now
    included, along with **past tense** (`I've sent that over`), which is the worse case: not an
    unkept promise but a completed action that never happened. `turn_had_tool` already excludes
    legitimate post-tool read-backs.
  - When adding a tool, **add its verbs here too.** The gate that stops the model inventing the
    action is this regex, not the prompt.

**A blank credential must not become an HTTP request (2026-09-04).** Same trace, t=44.8. The
caller's opening sentence named them, so the model called `verify_identity` immediately with
`date_of_birth: ""`. The API correctly 422'd, but a raw `Client error '422 Unprocessable'` is
not something a model can recover from gracefully — it apologized to the caller for a mistake
they could not perceive (*"I apologize — let me ask that differently"*) and spent a turn.
`_MISSING_DOB_RESULT` now answers it in the reducer with the instruction, the same way the
identity gate does, one step earlier and with no round trip.

**The Tier-A load test measured NOTHING from Phase 13 until 2026-09-04, and exited 0 the whole
time.** Phase 13 added `context_note` to `AnthropicLLM.start()`; `loadtest/fake_adapters.FakeLLM`
was never updated, so every session raised `unexpected keyword argument 'context_note'` on its
first turn. The harness printed a tidy table of `None`s with `*undersampled`, wrote an SVG, and
returned success — 800 of 800 sessions dead. Both halves are fixed and both matter:

  - `FakeLLM.start` now mirrors the real signature **keyword for keyword, spelled out rather
    than swallowed by `**kwargs`** — the next drift must fail loudly at the seam, not be
    silently accepted.
  - `tier_a.py` collects `worker.stats.failed`, prints `*** N SESSIONS FAILED ***`, and
    **returns 1**. Undersampling is a warning; sessions dying is a failure.

  The Phase-11 numbers in this file predate the break and are still valid. Any load-test result
  produced between Phase 13 and 2026-09-04 is not.

**Replaying a trace through a CHANGED reducer diverges, and that is not a bug.** Request ids are
derived from state counters, so a fix that changes which requests exist renumbers everything
after it and later deltas read as stale. `20260904T173146047919Z` replays to 2 utterances
instead of 15 for exactly this reason (the nudge no longer fires on its first turn). When using
replay as a regression check, diff a trace whose recorded engine matches the code under test.

**Billing blocker CLEARED (2026-09-03).** Credits topped up; the previous
`credit balance is too low` failure is resolved and live calls run end to end again.

Note for whoever runs the promptfoo suite: `eval/promptfoo/run.sh` passes `--no-cache`, so every
run is 72 live model calls plus rubric grading. That was right while the prompt was changing;
for routine runs, drop the flag and let promptfoo replay unchanged cases from cache.

**Live-testing loop that works:** place the call, then
`agent/.venv/bin/python agent/scripts/inspect_call.py` — it prints the caller turns, every tool
call and whether it succeeded, and a verdict line. `NO TOOL CALLS AT ALL` or
`booking committed: no` after the caller heard a confirmation is the tell for a fabricated
booking. Console `[media]` lines say whether caller audio arrived at all.

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

**Phase 5 — telephony (done)** (LiveKit SIP, `+14842951203`): `MODE` switch in
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
