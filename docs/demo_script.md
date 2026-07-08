# Demo Recording Guide

A short, repeatable script for recording the demo call linked from the README. The agent runs on
a real phone number (`+1 484 295 0169`), so a recording is one QuickTime capture away — no staging
environment needed.

Record **three short scenarios**: the happy path first (it's the money shot), then no-availability,
then a change of mind. Keep each call under **90 seconds**. All patient data below is synthetic.

---

## Before you record

1. **Start the stack** so the dashboard is live and metrics are captured:
   ```bash
   ./start_demo.sh          # API + agent (MODE=telephony); prints the dashboard URL
   ```
   Wait for the agent to log that it has joined room `clinic-inbound` and is waiting for a caller.

2. **Arrange the screen** so the recording shows both sides of the system at once:
   - **Left:** the terminal running `start_demo.sh` — this streams the live stage logs
     (`ASR ▶ / LLM ▶ / TOOL ▶ / TTS ▶`) as you talk, and prints the `[session]` summary line the
     moment you hang up.
   - **Right (optional but great):** a browser on `/dashboard/` — the latency table, ASR
     confidence, tool success, and outcomes refresh every 5 s, so the call shows up live.

3. **Start the QuickTime screen recording *with audio*** (see recording tips below), then place the
   call from your phone on speaker so both your voice and the agent's replies are captured.

---

## Scenario 1 — Happy path (record this first)

**Goal:** greeting + AI disclosure → intake → offer → confirm → booking with a confirmation number.

| You (caller) | Agent (expected) |
|--------------|------------------|
| *(call connects, listen to the greeting)* | "Thanks for calling Grove Family Clinic. You're speaking with an automated AI assistant. This call may be recorded for quality and scheduling purposes. I can help you book an appointment. How can I help today?" |
| "Hi, I'd like to book an appointment." | Asks for your name. |
| "Jane Doe." | Asks for your date of birth. |
| "March 4th, 1990." | Asks whether you're a new or existing patient. |
| "New patient." | Asks the reason for the visit. |
| "Just an annual checkup." | Offers one or two concrete open slots. |
| "Tuesday at 9 works." | Reads the appointment back and asks you to confirm. |
| "Yes, book it." | Confirms with a **confirmation number** and closes the call. |

**What to point out on screen:** as you speak, the terminal shows `ASR ▶ … LLM ▶ … TOOL ▶ check_availability … TTS ▶ …`. When you confirm, watch for `TOOL ▶ confirm_booking` followed by the confirmation number. On hangup, the `[session]` line prints the per-call latency summary:
```
[session] outcome=booked turns=… | ASR p50=…ms | LLM p50=…ms | TTS p50=…ms | E2E p50=…ms
```

## Scenario 2 — No availability

**Goal:** show the agent handling an empty window honestly instead of inventing a slot.

| You (caller) | Agent (expected) |
|--------------|------------------|
| "I'd like to book an appointment." | Runs intake as above (name, DOB, patient status, reason). |
| *(provide the intake details)* | Offers open slots. |
| "Do you have anything in August?" | Explains it doesn't have slots that far out and offers the **soonest available** instead. |
| "That's fine, book the earliest." | Reads it back, confirms, books. |

**What to point out:** the mock API seeds ~3 working days of slots, so a far-future request has
nothing to offer — the agent falls back to the soonest slot rather than failing. This is the
intended behavior, not a bug.

## Scenario 3 — Change of mind

**Goal:** show barge-in / mid-flow correction — the caller changes the day after an offer.

| You (caller) | Agent (expected) |
|--------------|------------------|
| "I'd like to book an appointment." | Runs intake. |
| *(provide the intake details, reason "sore throat")* | Offers slots for the first day. |
| "Actually, can we do the next day instead?" | Re-queries the new day and offers concrete times. |
| "The morning one." | Reads it back, confirms. |
| "Yes." | Books and closes with a confirmation number. |

**What to point out:** you can interrupt the agent mid-sentence (barge-in) and it re-queries the new
day, then binds your choice to a concrete offered slot before booking — no loop, one confirmation.

---

## Recording tips

- **Use QuickTime screen recording with audio.** On macOS: QuickTime Player → File → New Screen
  Recording → click the arrow next to the record button and select your microphone so both your
  voice and the phone speaker are captured. (Put the phone on speaker.)
- **Do the happy path first** — it's the clearest and most compelling; lead the final cut with it.
- **Keep each call under 90 seconds.** Speak naturally but don't over-explain; the agent is fast to
  respond except for the ~3 s LLM turn, which is worth letting the viewer see honestly.
- **Frame both panels** (terminal + dashboard) so the recording proves the observability story, not
  just the audio.
- Do a throwaway practice call first to settle audio levels and the greeting timing.
- Trim dead air at the start/end in QuickTime (Edit → Trim) before exporting.

## Where the recording goes

Upload the final cut (YouTube unlisted, Loom, or a repo asset) and replace the placeholder in the
README's **Try it** section:

```markdown
> **[Demo video]** — _(recording link — see docs/demo_script.md)_
```

Swap `[Demo video]` for a real markdown link to the recording, e.g.
`> **[▶ Demo video](https://…)**`.
