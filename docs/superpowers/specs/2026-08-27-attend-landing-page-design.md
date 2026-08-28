# Attend — landing page design

**Date:** 2026-08-27
**Status:** approved shape, spec under review
**Scope:** one static marketing page for the voice-agent product built in this repo

## Purpose

A single landing page that reads as a real product site for a clinic voice-agent company,
framed honestly as a working prototype. Two audiences, one page:

- **Recruiters / hiring managers** (primary) — see product sense *and* engineering rigor.
- **Clinic operators** (secondary) — understand the pain, the fix, and what it costs them to try.

Success: a visitor understands what it does in ten seconds, calls the number in sixty, and can
verify every claim on the page without taking anything on faith.

## Name and positioning

**Attend.** *Attending* is native healthcare vocabulary and *to attend* is what a front desk
does — be present, answer, take care of someone. It carries no clinical-authority implication,
which matters: the agent takes a history, it does not practise medicine.

Tagline: **The front desk that never misses a call.**

Narrative order — problem, product, proof:

1. **Problem.** Clinics lose booked revenue to unanswered phones: voicemail at peak, nothing
   after 5pm, and a front desk that cannot be in two conversations at once.
2. **Product.** Most voice AI can *talk*. Very little of it can *transact*. Attend holds a real
   slot, writes to a real database, and returns a real confirmation number.
3. **Proof.** Every capability claim on the page is backed by a number produced by the repo,
   and the page says where each number came from — including the ones that are unflattering.

The honesty is the differentiator, not a disclaimer in the footer. The strongest single line on
the page is that the agent is built so it *cannot* claim a booking it did not make — because
that failure was observed on a live call and fixed, and the page shows the trace.

## Page structure

One page, one CTA, no scroll traps.

| # | Section | Content | Why it earns its place |
|---|---|---|---|
| 1 | Hero | Name, tagline, one-sentence description, **phone number as the primary CTA**, one line setting expectation ("a real agent answers; it books into a real calendar") | The callable demo is the rarest thing here |
| 2 | The problem | Three short stats/claims on missed calls and after-hours demand, framed as an operator's day | Establishes why this exists before saying what it is |
| 3 | How it works | Three steps: caller speaks → agent collects intake and offers real open slots → booking is written and a confirmation number is read back. Plain language, no stack names | An operator must understand it without engineering vocabulary |
| 4 | **Live call replay** | Animated turn-by-turn replay of a real booking, transcript on the left, tool calls firing on the right with their real arguments and results | Shows the mechanism instead of asserting it. The memorable element |
| 5 | Proof | Measured numbers table + what each was measured on. Eval, safety, concurrency, latency | Every claim above, made falsifiable |
| 6 | What this is / isn't | Prototype status, synthetic data, no PHI, no BAAs, what would be required to take real patient calls | Turns the biggest weakness into the most credible thing on the page |
| 7 | Footer | Repo link, who built it, the demo-line caveat | — |

Explicitly out of scope: pricing, testimonials, team bios, blog, signup form, newsletter,
cookie banner. Nothing invented, nothing that collects data.

## The live call replay (section 4)

The one piece of real interaction design on the page.

**Data source.** `eval/results/20260813T045758Z.json`, case `hp_checkup_tomorrow` — a real run
against the real scheduling API on Claude Haiku 4.5, containing nine caller utterances, the
assistant's turns, and three tool calls with their genuine arguments and results (ending in
confirmation number `3BA83DD4`).

**Honesty constraint — load-bearing.** This is a *text-in-the-loop eval case, not a phone call*,
and the section must label it as such. The repo does not yet contain a trace of a successful
booking placed over the phone through the in-house engine: call 1 was deaf, call 2 made zero
tool calls. Presenting eval output as a recorded phone call would be exactly the fabrication
this page claims the product prevents.

**Upgrade path.** When the verification call lands (see Dependencies), swap the data for that
call's `logs/traces/<call_id>.jsonl` and relabel the section as a real phone call. The component
reads a trace-shaped JSON array, so this is a data swap, not a rewrite.

**Behavior.**
- Auto-plays once on scroll into view; pause/replay control; respects `prefers-reduced-motion`
  by rendering the whole transcript statically.
- Left column: transcript, caller and agent turns styled distinctly.
- Right column: tool calls appear as they fire — name, arguments, and the salient part of the
  result (slot count, hold id shortened, confirmation number).
- The `confirm_booking` call is the visual climax: the confirmation number appears in the tool
  panel *before* the agent speaks it, which is the whole architectural point made visually.
- Data embedded inline as a JS constant. No fetch — the page must work from `file://`.

## Visual direction

Calm, clinical, and confident; not a purple-gradient AI startup page.

- **Type:** one humanist sans for UI, and a mono for anything that is machine output
  (tool calls, numbers, the confirmation code). The mono/sans split is what makes "these are
  receipts, not marketing" legible without a word of explanation.
- **Color:** a restrained ink-and-paper base with a single saturated accent used only for the
  phone CTA and the live-call state. Full light and dark palettes as CSS custom properties.
- **Motion:** only in the replay, and only to show sequence. No parallax, no scroll-jacking.
- **Layout:** single column, generous measure, max ~72ch for prose. The replay is the only
  two-column block, collapsing to stacked on narrow screens.

## Numbers to publish

Each with its provenance stated on the page. Numbers are pulled from the repo at build time by
hand and cited; none are estimated.

| Claim | Value | Measured on |
|---|---|---|
| Eval pass rate | 19/19 cases, 100% task completion | Structural tool-trace scoring, Claude Haiku 4.5 |
| Emergency recall | 100%, 0 false positives (61 cases) | Deterministic detector, no model in the path |
| Intent accuracy | 98.3% | 60 labeled utterances |
| Concurrency | 1,000 sessions/process (1,600 with VAD on every frame) | Tier-A synthetic load, one process |
| Recommended capacity | ~800 | Event-loop lag grows superlinearly past ~400 |
| Booking race safety | exactly 1 winner of 20 concurrent holds | Compare-and-swap; the pre-Phase-9 code let 9 of 20 win |
| ASR latency | 87 ms p50 | Live call, Deepgram |
| TTS latency | 134 ms p50 | Live call, Cartesia |
| End-to-end latency | 4.1 s p50 (Pipecat path) | Live call, Phase 6 |
| Automated tests | 243 agent, plus API and router suites | `uv run pytest` |

**Latency must be handled carefully.** The in-house engine's most recent live call logged E2E
p50 ≈ 1.35 s, roughly 3× better than the 4.1 s Pipecat figure — but every turn in that call made
zero tool calls, so it is not a valid booking-turn number. The page publishes the 4.1 s figure
with the LLM named as the bottleneck, and states that the rebuilt engine measures faster with a
tool-call-inclusive number still outstanding. Publishing 1.35 s unqualified would be the same
category of error as the fabricated confirmation number.

## Technical approach

- **One self-contained HTML file.** Inline CSS and JS, no build step, no dependencies, no
  external requests. Opens from `file://`, deploys anywhere static.
- **Location:** `site/index.html` in the repo — source of truth, deployable to any static host.
- **Preview:** published as a Claude Artifact for a shareable private link. The repo file is
  canonical; the artifact is a redeployable view of it.
- **Responsive:** single column, fluid type, replay collapses to stacked below ~840px.
- **Theme:** light and dark via CSS custom properties, honoring `prefers-color-scheme`.
- **Accessibility:** semantic landmarks, the phone CTA is a real `tel:` link, replay is
  pausable, reduced-motion renders statically, contrast ≥ WCAG AA in both themes.

## Verification

- Page renders with no console errors and makes zero network requests (DevTools network tab
  empty).
- Every number on the page traces to a source named in the table above; a reviewer can check
  each one against the repo.
- Replay: plays, pauses, replays, and renders complete under `prefers-reduced-motion`.
- Layout holds at 360px, 768px, and 1440px with no horizontal scroll.
- Both themes legible; `tel:` link opens the dialer on a phone.

## Dependencies and pre-publish gate

The page is buildable now. **Publishing it publicly is gated** on three things that are not
page work:

1. **One clean verified live call** — a booking placed by phone through the in-house engine and
   confirmed present in the database. Until this exists, the CTA points at a number whose last
   two calls failed (deaf, then fabricated). This also supplies the real trace for section 4.
2. **A spend guard** — the number on a public page has no rate limit and no cap across four paid
   APIs. Publishing without one is an open invoice.
3. **Supabase kept warm** — the free tier pauses after ~7 days idle, and a paused database means
   the agent cannot book. A public CTA pointing at a sleeping database is a dead demo.

The user has stated they will not publish until the work is complete, so these are a checklist
rather than a blocker on building.

## Open questions

None blocking. The trace swap in section 4 is planned work, not an unknown.
