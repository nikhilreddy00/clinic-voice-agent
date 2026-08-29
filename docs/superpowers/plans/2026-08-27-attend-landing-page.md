# Attend Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained landing page for **Attend**, the clinic voice-agent product in this repo, whose every claim is backed by a number the repo can produce.

**Architecture:** One static HTML file at `site/index.html` with inline CSS and JS — no build step, no dependencies, no network requests, opens from `file://`. A Python check script asserts the page's invariants (no external references, real phone number, published numbers present, accessibility affordances). A second Python script regenerates the call-replay data block from a real result file, so swapping in a live phone trace later is one command.

**Tech Stack:** Plain HTML5, CSS custom properties, vanilla ES2020. Python 3.12 stdlib for the two scripts. No frameworks, no fonts loaded over the network, no package manager.

**Spec:** `docs/superpowers/specs/2026-08-27-attend-landing-page-design.md`

## Global Constraints

- **Zero network requests.** No CDN, no Google Fonts, no remote images, no `fetch`. All assets inline. This is asserted by `site/check_page.py` and is non-negotiable — it is what lets the page open from `file://` and deploy anywhere.
- **Product name is `Attend`.** Tagline verbatim: `The front desk that never misses a call.`
- **Demo clinic name is `Grove Family Clinic`.** Attend is the product; Grove is the clinic it serves. Never conflate them.
- **Phone number:** display `+1 (484) 295-0169`, link `tel:+14842950169`.
- **No invented facts.** No pricing, no testimonials, no customer names, no team bios, no metrics not listed in the "Numbers" table of the spec.
- **The replay is labeled a text-in-the-loop eval case, not a phone call.** Wording must not imply recorded audio or a phone conversation.
- **All patient data is synthetic** and the page says so wherever patient-shaped data appears.
- **Latency published is 4.1 s p50** (Pipecat path) with the LLM named as the bottleneck. Do NOT publish the ~1.35 s in-house-engine figure as a booking-turn number.
- **Theme:** light and dark, both defined via CSS custom properties on `:root`, dark under `@media (prefers-color-scheme: dark)`.
- **Reduced motion:** under `prefers-reduced-motion: reduce`, the replay renders complete and static.

---

### Task 1: Check script, design tokens, and hero

Establishes the file, the invariant checker that every later task re-runs, and the first visible section.

**Files:**
- Create: `site/check_page.py`
- Create: `site/index.html`

**Interfaces:**
- Consumes: nothing.
- Produces: `site/index.html` containing the CSS custom-property block (`--bg`, `--fg`, `--muted`, `--line`, `--accent`, `--accent-fg`, `--surface`, `--mono-bg`) that all later tasks style against; `site/check_page.py` exposing `main() -> int` and run as `python3 site/check_page.py`.

- [ ] **Step 1: Write the failing check script**

Create `site/check_page.py`:

```python
#!/usr/bin/env python3
"""Invariant checks for the Attend landing page.

The page has no build step and no JS test framework — adding one to a Python repo for a
single static file would cost more than it catches. These are the assertions that actually
matter for this page: it must make no network requests, it must carry the real phone number,
and every number it publishes must be one the repo can back up.

Run: python3 site/check_page.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).parent / "index.html"

# Anything that would make the browser reach off-origin. The page must be openable from
# file:// with the network unplugged.
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["'](?:https?:)?//""" r"""|@import\s+url\(["']?https?:""" r"""|fetch\s*\(""",
    re.IGNORECASE,
)

REQUIRED_STRINGS = [
    "Attend",
    "The front desk that never misses a call.",
    "Grove Family Clinic",
    'href="tel:+14842950169"',
    "+1 (484) 295-0169",
]


def check(html: str) -> list[str]:
    """Return a list of failures; empty means the page is good."""
    failures: list[str] = []

    for hit in EXTERNAL.findall(html):
        failures.append(f"external reference found: {hit!r}")

    for needle in REQUIRED_STRINGS:
        if needle not in html:
            failures.append(f"missing required string: {needle!r}")

    if "prefers-color-scheme: dark" not in html:
        failures.append("no dark theme block")
    if "prefers-reduced-motion" not in html:
        failures.append("no reduced-motion block")
    if "<title>" not in html:
        failures.append("no <title>")
    if 'lang="en"' not in html:
        failures.append("no lang attribute on <html>")

    return failures


def main() -> int:
    if not PAGE.exists():
        print(f"FAIL: {PAGE} does not exist")
        return 1
    failures = check(PAGE.read_text(encoding="utf-8"))
    if failures:
        print(f"FAIL ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {PAGE.name} passed all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 site/check_page.py`
Expected: `FAIL: .../site/index.html does not exist`, exit code 1.

- [ ] **Step 3: Create the page skeleton, tokens, and hero**

Create `site/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attend — the front desk that never misses a call</title>
<meta name="description" content="A voice agent that answers a clinic's phone, takes a real intake, and books into a real calendar. Built as a working prototype, with the measurements to back it up.">
<style>
  :root {
    --bg: #FBFAF8;
    --surface: #FFFFFF;
    --fg: #16181D;
    --muted: #5B6270;
    --line: #E4E1DB;
    --accent: #0F5C4E;
    --accent-fg: #FFFFFF;
    --mono-bg: #F3F1EC;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    --measure: 68ch;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0E1013;
      --surface: #14171C;
      --fg: #ECEEF2;
      --muted: #99A1B0;
      --line: #232830;
      --accent: #35C7A9;
      --accent-fg: #06231D;
      --mono-bg: #171B21;
    }
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--sans);
    font-size: clamp(1rem, 0.96rem + 0.2vw, 1.0625rem);
    line-height: 1.6;
  }
  main { max-width: 78rem; margin: 0 auto; padding: 0 1.5rem; }
  section { padding: clamp(3rem, 8vw, 6rem) 0; border-top: 1px solid var(--line); }
  section:first-of-type { border-top: 0; }
  h1, h2, h3 { line-height: 1.15; letter-spacing: -0.02em; margin: 0 0 0.75rem; }
  h1 { font-size: clamp(2.25rem, 1.6rem + 3.2vw, 4rem); }
  h2 { font-size: clamp(1.5rem, 1.2rem + 1.4vw, 2.25rem); }
  p { max-width: var(--measure); margin: 0 0 1rem; }
  .lede { font-size: clamp(1.125rem, 1rem + 0.6vw, 1.375rem); color: var(--muted); }
  .eyebrow {
    font: 600 0.75rem/1 var(--mono);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
  }
  code, .mono { font-family: var(--mono); font-size: 0.9em; }

  /* --- hero ------------------------------------------------------------------ */
  .hero { padding-top: clamp(3rem, 10vw, 7rem); }
  .wordmark {
    font: 700 1.125rem/1 var(--sans);
    letter-spacing: -0.02em;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: clamp(2rem, 6vw, 4rem);
  }
  .wordmark::before {
    content: "";
    width: 0.7rem; height: 0.7rem; border-radius: 50%;
    background: var(--accent);
  }
  .cta {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 1rem 1.5rem;
    margin-top: 2.5rem;
  }
  .call {
    display: inline-flex; align-items: baseline; gap: 0.6rem;
    background: var(--accent); color: var(--accent-fg);
    text-decoration: none;
    padding: 0.9rem 1.4rem;
    border-radius: 0.6rem;
    font: 600 1.125rem/1 var(--sans);
  }
  .call .num { font-family: var(--mono); letter-spacing: -0.01em; }
  .call:hover { filter: brightness(1.08); }
  .call:focus-visible { outline: 3px solid var(--fg); outline-offset: 3px; }
  .cta-note { color: var(--muted); font-size: 0.9375rem; max-width: 34ch; margin: 0; }
</style>
</head>
<body>
<main>

  <section class="hero">
    <div class="wordmark">Attend</div>
    <h1>The front desk that never misses a call.</h1>
    <p class="lede">
      Attend answers a clinic's phone, takes a real intake, offers times that are genuinely
      open, and books the appointment — then reads back a confirmation number it actually has.
    </p>
    <div class="cta">
      <a class="call" href="tel:+14842950169">
        Call the demo line <span class="num">+1 (484) 295-0169</span>
      </a>
      <p class="cta-note">
        A real agent picks up and books into Grove Family Clinic, a synthetic practice used
        for demonstration.
      </p>
    </div>
  </section>

</main>
</body>
</html>
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `python3 site/check_page.py`
Expected: `OK: index.html passed all checks`, exit code 0.

- [ ] **Step 5: Open it and confirm it renders**

Run: `open site/index.html`
Expected: hero renders, the phone button is a real link, no console errors, and DevTools → Network shows zero requests.

- [ ] **Step 6: Commit**

```bash
git add site/index.html site/check_page.py
git commit -m "feat(site): Attend landing page skeleton, tokens, and hero"
```

---

### Task 2: The problem and how it works

**Files:**
- Modify: `site/index.html` (append two `<section>` blocks before `</main>`; append CSS before `</style>`)
- Modify: `site/check_page.py` (extend `REQUIRED_STRINGS`)

**Interfaces:**
- Consumes: the CSS custom properties and `section` / `h2` / `.eyebrow` / `.lede` rules from Task 1.
- Produces: `.cols` and `.steps` CSS classes reused by Task 4.

- [ ] **Step 1: Extend the check with the new required copy**

In `site/check_page.py`, add to `REQUIRED_STRINGS`:

```python
    "Three in ten calls to a busy practice go unanswered",
    "It cannot tell a caller they are booked unless they are.",
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `python3 site/check_page.py`
Expected: FAIL, listing both missing strings.

- [ ] **Step 3: Add the CSS**

Insert before `</style>` in `site/index.html`:

```css
  /* --- generic layout blocks -------------------------------------------------- */
  .cols {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
    gap: clamp(1.5rem, 4vw, 3rem);
    margin-top: 2.5rem;
  }
  .cols h3 { font-size: 1.0625rem; }
  .cols p { color: var(--muted); margin: 0; font-size: 0.9375rem; }
  .stat {
    font: 700 clamp(2rem, 1.4rem + 2.4vw, 3rem)/1 var(--sans);
    letter-spacing: -0.03em;
    color: var(--accent);
    display: block;
    margin-bottom: 0.35rem;
  }
  .steps { counter-reset: step; margin-top: 2.5rem; padding: 0; list-style: none; }
  .steps li {
    counter-increment: step;
    display: grid;
    grid-template-columns: 2.25rem 1fr;
    gap: 1rem;
    padding: 1.25rem 0;
    border-top: 1px solid var(--line);
  }
  .steps li::before {
    content: counter(step);
    font: 600 0.875rem/2.25rem var(--mono);
    text-align: center;
    color: var(--accent-fg);
    background: var(--accent);
    border-radius: 50%;
    width: 2.25rem; height: 2.25rem;
  }
  .steps h3 { margin-bottom: 0.25rem; font-size: 1.0625rem; }
  .steps p { color: var(--muted); margin: 0; }
```

- [ ] **Step 4: Add the two sections**

Insert before `</main>` in `site/index.html`:

```html
  <section id="problem">
    <p class="eyebrow">The problem</p>
    <h2>The phone is the front door, and it is often locked.</h2>
    <p class="lede">
      Most practices lose booked appointments the same way: nobody could pick up.
    </p>
    <div class="cols">
      <div>
        <span class="stat">3 in 10</span>
        <h3>Calls go unanswered at peak</h3>
        <p>
          Three in ten calls to a busy practice go unanswered when the desk is with a patient,
          on another line, or at lunch. Each one is a patient who has to try again — or doesn't.
        </p>
      </div>
      <div>
        <span class="stat">After 5</span>
        <h3>The line goes dark</h3>
        <p>
          Patients decide to book in the evening, which is exactly when nobody is there to
          take the call. Voicemail is where that intention goes to expire.
        </p>
      </div>
      <div>
        <span class="stat">1 at a time</span>
        <h3>A desk cannot parallelize</h3>
        <p>
          One receptionist holds one conversation. A Monday-morning rush is a queue by
          definition, and callers who wait on hold hang up.
        </p>
      </div>
    </div>
  </section>

  <section id="how">
    <p class="eyebrow">How it works</p>
    <h2>It answers, it asks the right questions, and it books.</h2>
    <ol class="steps">
      <li>
        <div>
          <h3>It picks up — every time, on every line</h3>
          <p>
            The caller hears that they are speaking to an automated assistant, and that the
            call is recorded, before they say anything. That disclosure is not optional.
          </p>
        </div>
      </li>
      <li>
        <div>
          <h3>It takes an intake a clinician can actually use</h3>
          <p>
            Name, date of birth, new or returning — then a focused history: how long, how bad,
            what makes it worse, what's happened before. It asks a few targeted questions and
            writes one clean clinical sentence, so nobody re-takes the history in the room.
            It never diagnoses and never suggests treatment.
          </p>
        </div>
      </li>
      <li>
        <div>
          <h3>It books into the real calendar</h3>
          <p>
            It offers only times the schedule actually has open, holds the slot while the
            caller decides, writes the booking, and reads back the confirmation number the
            system returned. It cannot tell a caller they are booked unless they are.
          </p>
        </div>
      </li>
    </ol>
  </section>
```

- [ ] **Step 5: Run the check to verify it passes**

Run: `python3 site/check_page.py`
Expected: `OK`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add site/index.html site/check_page.py
git commit -m "feat(site): problem and how-it-works sections"
```

---

### Task 3: The live call replay

The one interactive element. Data comes from a real eval run; the generator makes the future swap to a live phone trace a single command.

**Files:**
- Create: `site/build_trace.py`
- Modify: `site/index.html` (replay section, CSS, and the generated data block)
- Modify: `site/check_page.py` (assert the sentinels and the label)

**Interfaces:**
- Consumes: CSS tokens from Task 1.
- Produces: in `site/index.html`, a JS constant `CALL_REPLAY` — an array of step objects, each
  `{"role": "caller"|"agent", "text": str}` or `{"role": "tool", "name": str, "args": object, "result": object}` —
  bounded by the exact sentinel comments `/* BEGIN GENERATED TRACE */` and `/* END GENERATED TRACE */`.
  `site/build_trace.py` rewrites everything between those sentinels in place.

- [ ] **Step 1: Write the trace generator**

Create `site/build_trace.py`:

```python
#!/usr/bin/env python3
"""Regenerate the landing page's call-replay data from a real result file.

Why a generator instead of pasting JSON once: the numbers and the confirmation code on that
page are the whole argument, and retyping them by hand is how a wrong one ships. This derives
them from the source file every time.

It also makes the planned swap cheap. Today the replay uses a text-in-the-loop eval case,
because no successful booking has yet been placed over the phone through the in-house engine.
When one is, point this at that call's trace and rerun.

Run: python3 site/build_trace.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).parent / "index.html"
SOURCE = ROOT / "eval" / "results" / "20260813T045758Z.json"
CASE_ID = "hp_checkup_tomorrow"

BEGIN = "/* BEGIN GENERATED TRACE */"
END = "/* END GENERATED TRACE */"

# The eval result records utterances, assistant turns, and tool calls as three flat lists with
# no turn index linking them — the harness never needed one. These positions are INFERRED from
# the dialogue and corroborated by the assistant turns: availability is checked after the
# caller says any time works (turn 7), the slot is held when they accept (turn 8), and the
# booking is committed after the explicit yes (turn 9). The page therefore shows sequence and
# never a per-turn timestamp, because the source has none to show.
TOOL_AFTER_CALLER_TURN = {7: "check_availability", 8: "hold_slot", 9: "confirm_booking"}


def build_steps() -> list[dict]:
    case = next(
        c for c in json.loads(SOURCE.read_text())["cases"] if c["id"] == CASE_ID
    )
    by_name = {t["name"]: t for t in case["tool_calls"]}
    steps: list[dict] = []

    for i, (caller, agent) in enumerate(
        zip(case["utterances"], case["assistant_turns"]), start=1
    ):
        steps.append({"role": "caller", "text": caller})
        tool_name = TOOL_AFTER_CALLER_TURN.get(i)
        if tool_name and tool_name in by_name:
            tool = by_name[tool_name]
            steps.append(
                {
                    "role": "tool",
                    "name": tool["name"],
                    "args": tool.get("args") or {},
                    "result": summarize(tool["name"], tool.get("result") or {}),
                }
            )
        steps.append({"role": "agent", "text": agent})
    return steps


def summarize(name: str, result: dict) -> dict:
    """Keep the part of a tool result a viewer can read in two seconds.

    confirm_booking's stored result carries null date_of_birth / new_patient / symptom_notes —
    a known eval-harness defect where those args are sent but not persisted. Showing nulls
    would misrepresent the call, so the summary takes only fields that are actually populated.
    """
    if name == "check_availability":
        return {"ok": result.get("ok"), "slots_found": result.get("count")}
    if name == "hold_slot":
        held = str(result.get("hold_id", ""))
        return {"ok": result.get("ok"), "hold_id": held[:8] + "…" if held else None}
    if name == "confirm_booking":
        return {
            "ok": result.get("ok"),
            "confirmation_id": result.get("confirmation_id"),
            "display_time": result.get("display_time"),
            "provider_name": result.get("provider_name"),
        }
    return {"ok": result.get("ok")}


def main() -> int:
    if not SOURCE.exists():
        print(f"FAIL: source not found: {SOURCE}")
        return 1
    html = PAGE.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        print(f"FAIL: sentinels not found in {PAGE}")
        return 1

    block = "const CALL_REPLAY = " + json.dumps(build_steps(), indent=2) + ";"
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL
    )
    PAGE.write_text(
        pattern.sub(f"{BEGIN}\n{block}\n{END}", html), encoding="utf-8"
    )
    print(f"OK: regenerated trace from {SOURCE.name} ({CASE_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Extend the check script**

In `site/check_page.py`, add to `REQUIRED_STRINGS`:

```python
    "/* BEGIN GENERATED TRACE */",
    "const CALL_REPLAY",
    "3BA83DD4",
    "text-in-the-loop eval case, not a phone call",
```

- [ ] **Step 3: Run the check to verify it fails**

Run: `python3 site/check_page.py`
Expected: FAIL, listing the four missing strings.

- [ ] **Step 4: Add the replay CSS**

Insert before `</style>` in `site/index.html`:

```css
  /* --- call replay ------------------------------------------------------------ */
  .replay {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
    gap: clamp(1rem, 3vw, 2rem);
    margin-top: 2.5rem;
    align-items: start;
  }
  @media (max-width: 840px) { .replay { grid-template-columns: 1fr; } }
  .panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0.75rem;
    padding: 1.25rem;
    min-height: 22rem;
  }
  .panel-title {
    font: 600 0.75rem/1 var(--mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex; justify-content: space-between; align-items: center;
  }
  .turn { margin-bottom: 0.875rem; opacity: 0; transform: translateY(4px); }
  .turn.shown { opacity: 1; transform: none; transition: opacity .28s ease, transform .28s ease; }
  .turn .who {
    font: 600 0.6875rem/1 var(--mono);
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); display: block; margin-bottom: 0.25rem;
  }
  .turn.caller .bubble { background: var(--mono-bg); }
  .turn.agent .bubble { background: transparent; border: 1px solid var(--line); }
  .bubble { padding: 0.6rem 0.8rem; border-radius: 0.5rem; }
  .tool {
    font-family: var(--mono); font-size: 0.8125rem;
    border: 1px solid var(--line); border-left: 3px solid var(--accent);
    border-radius: 0.5rem; padding: 0.7rem 0.85rem; margin-bottom: 0.75rem;
    background: var(--mono-bg);
    opacity: 0; transform: translateY(4px);
  }
  .tool.shown { opacity: 1; transform: none; transition: opacity .28s ease, transform .28s ease; }
  .tool .name { color: var(--accent); font-weight: 600; }
  .tool dl { margin: 0.5rem 0 0; display: grid; grid-template-columns: auto 1fr; gap: 0.15rem 0.6rem; }
  .tool dt { color: var(--muted); }
  .tool dd { margin: 0; overflow-wrap: anywhere; }
  .tool.is-confirm { border-left-color: var(--accent); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 22%, transparent); }
  .replay-controls { display: flex; gap: 0.75rem; align-items: center; margin-top: 1.25rem; }
  .btn {
    font: 600 0.875rem/1 var(--sans);
    background: transparent; color: var(--fg);
    border: 1px solid var(--line); border-radius: 0.5rem;
    padding: 0.55rem 0.9rem; cursor: pointer;
  }
  .btn:hover { border-color: var(--accent); }
  .btn:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  .caveat { color: var(--muted); font-size: 0.875rem; max-width: var(--measure); }
  @media (prefers-reduced-motion: reduce) {
    .turn, .tool { opacity: 1; transform: none; transition: none; }
  }
```

- [ ] **Step 5: Add the replay markup and script**

Insert before `</main>` in `site/index.html`:

```html
  <section id="replay">
    <p class="eyebrow">See it work</p>
    <h2>A real booking, turn by turn.</h2>
    <p class="lede">
      Watch the tool calls fire on the right as the conversation happens on the left. The
      confirmation number appears in the system panel <em>before</em> the agent says it out
      loud — that ordering is the entire design.
    </p>
    <div class="replay">
      <div class="panel">
        <div class="panel-title"><span>Conversation</span></div>
        <div id="transcript"></div>
      </div>
      <div class="panel">
        <div class="panel-title"><span>System</span> <span id="tool-count">0 tool calls</span></div>
        <div id="tools"></div>
      </div>
    </div>
    <div class="replay-controls">
      <button class="btn" id="replay-btn" type="button">Replay</button>
    </div>
    <p class="caveat">
      This is a real run against the real scheduling API on Claude Haiku 4.5 — a
      text-in-the-loop eval case, not a phone call. Every tool call, argument and confirmation
      number shown is what the system actually produced. The patient is synthetic; all data in
      this project is. A recording of the same flow placed over the phone is coming once the
      rebuilt engine has one verified end to end.
    </p>
  </section>

<script>
/* BEGIN GENERATED TRACE */
const CALL_REPLAY = [];
/* END GENERATED TRACE */

(function () {
  const transcript = document.getElementById("transcript");
  const tools = document.getElementById("tools");
  const count = document.getElementById("tool-count");
  const button = document.getElementById("replay-btn");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let timers = [];
  let toolsSeen = 0;

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  function renderTurn(step) {
    const el = document.createElement("div");
    el.className = "turn " + step.role;
    el.innerHTML =
      '<span class="who">' + (step.role === "caller" ? "Caller" : "Attend") + "</span>" +
      '<div class="bubble">' + esc(step.text) + "</div>";
    transcript.appendChild(el);
    return el;
  }

  function renderTool(step) {
    const el = document.createElement("div");
    el.className = "tool" + (step.name === "confirm_booking" ? " is-confirm" : "");
    const rows = Object.entries(step.args)
      .concat(Object.entries(step.result))
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => "<dt>" + esc(k) + "</dt><dd>" + esc(v) + "</dd>")
      .join("");
    el.innerHTML = '<span class="name">' + esc(step.name) + "()</span><dl>" + rows + "</dl>";
    tools.appendChild(el);
    toolsSeen += 1;
    count.textContent = toolsSeen + (toolsSeen === 1 ? " tool call" : " tool calls");
    return el;
  }

  function reset() {
    timers.forEach(clearTimeout);
    timers = [];
    transcript.innerHTML = "";
    tools.innerHTML = "";
    toolsSeen = 0;
    count.textContent = "0 tool calls";
  }

  function play() {
    reset();
    let delay = 0;
    CALL_REPLAY.forEach((step) => {
      const render = step.role === "tool" ? renderTool : renderTurn;
      if (reduced) {
        render(step).classList.add("shown");
        return;
      }
      timers.push(
        setTimeout(() => {
          const el = render(step);
          requestAnimationFrame(() => el.classList.add("shown"));
          el.scrollIntoView({ block: "nearest" });
        }, delay)
      );
      delay += step.role === "tool" ? 700 : 1100;
    });
  }

  button.addEventListener("click", play);

  if (reduced) {
    play();
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          play();
          io.disconnect();
        }
      },
      { threshold: 0.25 }
    );
    io.observe(document.getElementById("replay"));
  }
})();
</script>
```

- [ ] **Step 6: Generate the real trace data**

Run: `python3 site/build_trace.py`
Expected: `OK: regenerated trace from 20260813T045758Z.json (hp_checkup_tomorrow)`, and `CALL_REPLAY` in the page is now a populated array containing `"3BA83DD4"`.

- [ ] **Step 7: Run the check to verify it passes**

Run: `python3 site/check_page.py`
Expected: `OK`, exit code 0.

- [ ] **Step 8: Verify the animation in a browser**

Run: `open site/index.html`
Expected: scrolling to the replay auto-plays it once; turns appear in order; three tool cards appear; `confirm_booking` is visually emphasized and its `confirmation_id` renders **before** the final agent turn is spoken; the Replay button restarts it; no console errors.

- [ ] **Step 9: Commit**

```bash
git add site/index.html site/build_trace.py site/check_page.py
git commit -m "feat(site): animated call replay driven by a real eval trace"
```

---

### Task 4: Proof, honest limits, and footer

**Files:**
- Modify: `site/index.html` (three sections + CSS)
- Modify: `site/check_page.py` (assert the published numbers and the honesty section)

**Interfaces:**
- Consumes: `.cols` from Task 2, tokens from Task 1.
- Produces: the final `</main>` content; no later task depends on new names.

- [ ] **Step 1: Extend the check with the published numbers**

In `site/check_page.py`, add to `REQUIRED_STRINGS`:

```python
    "19/19",
    "100%",
    "98.3%",
    "1,000",
    "4.1 s",
    "243",
    "What this is, and what it isn't",
    "No BAAs are signed",
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `python3 site/check_page.py`
Expected: FAIL, listing the missing strings.

- [ ] **Step 3: Add the table CSS**

Insert before `</style>` in `site/index.html`:

```css
  /* --- proof table ------------------------------------------------------------ */
  .table-wrap { overflow-x: auto; margin-top: 2.5rem; }
  table { border-collapse: collapse; width: 100%; min-width: 34rem; font-size: 0.9375rem; }
  th, td { text-align: left; padding: 0.7rem 1rem 0.7rem 0; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font: 600 0.75rem/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  td.value { font-family: var(--mono); color: var(--accent); font-weight: 600; white-space: nowrap; }
  td.source { color: var(--muted); }
  .limits { border-left: 3px solid var(--accent); padding-left: 1.25rem; margin-top: 2rem; }
  .limits li { margin-bottom: 0.6rem; max-width: var(--measure); }
  footer {
    border-top: 1px solid var(--line);
    padding: 2.5rem 0 4rem;
    color: var(--muted);
    font-size: 0.9375rem;
  }
  footer a { color: var(--fg); }
```

- [ ] **Step 4: Add the three sections**

Insert before `</main>` in `site/index.html`:

```html
  <section id="proof">
    <p class="eyebrow">Proof</p>
    <h2>Every claim above has a number behind it.</h2>
    <p class="lede">
      These are measurements this system produced, not projections. Where a number is
      unflattering, it is here anyway.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th scope="col">What</th><th scope="col">Measured</th><th scope="col">On what</th></tr>
        </thead>
        <tbody>
          <tr><td>Booking eval</td><td class="value">19/19</td><td class="source">Scripted and adversarial conversations, scored on the tool-call trace rather than the wording</td></tr>
          <tr><td>Emergency recall</td><td class="value">100%</td><td class="source">61 cases, zero false positives. Detection is deterministic — no model sits in that path</td></tr>
          <tr><td>Intent accuracy</td><td class="value">98.3%</td><td class="source">60 labeled caller utterances</td></tr>
          <tr><td>Concurrent calls</td><td class="value">1,000</td><td class="source">Per process, synthetic load, flat latency and zero timeouts — 1,600 with voice detection running on every frame. Recommended capacity is ~800, where loop lag starts climbing</td></tr>
          <tr><td>Double-booking</td><td class="value">0</td><td class="source">20 callers racing for one slot yield exactly one winner. The earlier read-then-write let 9 of 20 win</td></tr>
          <tr><td>Speech recognition</td><td class="value">87 ms</td><td class="source">Median, on a live call</td></tr>
          <tr><td>Speech synthesis</td><td class="value">134 ms</td><td class="source">Median, on a live call</td></tr>
          <tr><td>Voice to voice</td><td class="value">4.1 s</td><td class="source">Median on a live call. The language model is the bottleneck — roughly three of those four seconds. The rebuilt engine measures faster, but not yet on a call that made tool calls, so that number is not published here</td></tr>
          <tr><td>Automated tests</td><td class="value">243</td><td class="source">On the agent alone, plus separate suites for the scheduling API and the call router. Run on every change</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section id="limits">
    <p class="eyebrow">Straight answers</p>
    <h2>What this is, and what it isn't</h2>
    <p class="lede">
      Attend is a working prototype built to production shape. It is not a product you can buy
      today, and the gap is worth stating precisely.
    </p>
    <ul class="limits">
      <li>
        <strong>All data is synthetic.</strong> No real patient information has ever entered
        this system — not the database, not the logs, not the prompts.
      </li>
      <li>
        <strong>No BAAs are signed.</strong> Handling real patient data lawfully requires
        business associate agreements with every vendor in the path — telephony, speech
        recognition, the language model, speech synthesis, and hosting. That is an enterprise
        contract each, and until they exist this cannot take a real patient call. The
        architecture is built so that becoming compliant is configuration and contracts rather
        than a rewrite: every vendor sits behind a swappable adapter.
      </li>
      <li>
        <strong>It hands off rather than guessing.</strong> Refills, billing, test results and
        clinical questions are routed to staff, not improvised. An agent that confidently does
        something it cannot do is worse than one that says so.
      </li>
      <li>
        <strong>It cannot claim a booking it did not make.</strong> On an early call it did
        exactly that — narrated an appointment and invented a confirmation code after losing
        access to its own booking tools mid-conversation. That failure is why the rule now
        lives in the core instruction the agent carries on every single turn, and why the
        replay above shows the tool result landing before the agent speaks.
      </li>
    </ul>
  </section>

  <footer>
    <p>
      <strong>Attend</strong> — a clinic voice agent built as a working prototype.
      Grove Family Clinic is a synthetic practice used for demonstration.
    </p>
    <p>
      The demo line answers a real agent and books into a real database. Please be gentle with
      it; it is one person's prototype, not a service.
    </p>
  </footer>
```

- [ ] **Step 5: Run the check to verify it passes**

Run: `python3 site/check_page.py`
Expected: `OK`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add site/index.html site/check_page.py
git commit -m "feat(site): proof table, honest limits, and footer"
```

---

### Task 5: Responsive, accessibility, and theme verification

No new content. This is the pass that makes the page hold up, and it ends with the artifact published.

**Files:**
- Modify: `site/index.html` (fixes found during verification only)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: the finished page.

- [ ] **Step 1: Verify layout at three widths**

Open `site/index.html` and use DevTools device toolbar at **360px**, **768px**, and **1440px**.
Expected at every width: no horizontal page scroll; the replay is stacked below 840px; the
proof table scrolls inside its own container rather than widening the page; no text is clipped.
Fix any failure in the CSS before proceeding.

- [ ] **Step 2: Verify both themes**

In DevTools → Rendering → "Emulate prefers-color-scheme", check `light` then `dark`.
Expected: text remains legible in both, the accent stays visible against the background, and
the tool cards remain readable. Confirm body text against background clears WCAG AA (4.5:1)
using DevTools' contrast readout on a paragraph and on `.cta-note`.

- [ ] **Step 3: Verify reduced motion**

In DevTools → Rendering → "Emulate prefers-reduced-motion: reduce", reload.
Expected: the full transcript and all three tool cards are present immediately, with no
animation and no empty panels.

- [ ] **Step 4: Verify keyboard and semantics**

Tab through the page.
Expected: the phone link and the Replay button both receive a visible focus ring; the tab order
follows the visual order; every section has a heading; the page has exactly one `<h1>`.

- [ ] **Step 5: Verify zero network requests**

Open DevTools → Network, hard-reload the page.
Expected: no requests other than the document itself. Then run `python3 site/check_page.py`
one final time; expected `OK`.

- [ ] **Step 6: Commit**

```bash
git add site/index.html
git commit -m "fix(site): responsive, contrast, and reduced-motion pass"
```

- [ ] **Step 7: Publish the artifact**

Publish `site/index.html` as a Claude Artifact titled **Attend** with favicon `📞` and a
one-sentence description. Keep the repo file canonical; the artifact is a shareable view of it.
Report the URL.

---

## Pre-publish checklist (not part of the build)

The page is finished when Task 5 is done. Making the demo number public is gated on three
things from the spec, none of which are page work:

- [ ] One clean live call, verified present in the database, through the in-house engine — this
      also supplies the real phone trace to swap into the replay via `python3 site/build_trace.py`.
- [ ] A spend guard on the number: rate limiting and a cap across LiveKit, Deepgram, Anthropic
      and Cartesia.
- [ ] Supabase kept warm; the free tier pauses after ~7 days idle and a paused database means
      the agent cannot book.
