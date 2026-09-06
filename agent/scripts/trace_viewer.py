#!/usr/bin/env python3
"""Render one recorded call as a self-contained HTML page.

    agent/.venv/bin/python agent/scripts/trace_viewer.py               # newest call
    agent/.venv/bin/python agent/scripts/trace_viewer.py <call_id>
    agent/.venv/bin/python agent/scripts/trace_viewer.py <path> -o /tmp/call.html --open

`inspect_call.py` prints the same call and is still the right tool at a terminal. This exists
for the question a column of numbers answers badly: WHERE did a turn's two seconds go, and did
what the agent said match what it did. A waterfall answers that at a glance; a table does not.

IT DOES NOT LEAVE THE MACHINE. The page contains the caller's words, their name, and their date
of birth, because that is the whole point of reading a trace. It is written into git-ignored
`logs/` and is not an Artifact, not a deploy target, and not something to paste into a ticket.
The redacted view of the same call is the OTel trace (`core/otel.py`), which carries none of it.

BUILT OUT OF TWO THINGS THAT ALREADY EXIST, on purpose:

  * the waterfall is `core/otel.spans_from_events` — the same span tree the Grafana dashboard
    is built on. If the viewer and the dashboard ever disagree about what a turn cost, that is
    a bug in one shared mapper rather than a discrepancy between two hand-rolled ones;
  * the verdict is `eval/tier1_replay.check_events` — the same invariants CI runs on every
    commit. A call that violates one gets it printed at the top, in red, above the transcript.
"""

from __future__ import annotations

import argparse
import glob
import html
import os
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent" / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import tier1_replay as tier1  # noqa: E402

from clinic_agent.core import events as ev  # noqa: E402
from clinic_agent.core.otel import spans_from_events  # noqa: E402
from clinic_agent.core.recorder import load_trace, trace_dir  # noqa: E402

# Stage colours, shared by the waterfall bars and the legend. Ordered by where the time
# usually goes: LLM first, because on this stack it is ~50% of every turn.
STAGE_COLORS = {
    "llm": "#e0a33e",
    "stt": "#4f9ecf",
    "classifier": "#8f7bd4",
    "playback": "#4aa06b",
    "tool": "#cf5f5f",
}


def resolve(arg: str | None) -> Path:
    """A path, a call id, or nothing at all (newest trace in logs/traces)."""
    if arg:
        candidate = Path(arg)
        if candidate.exists():
            return candidate
        named = trace_dir() / f"{arg}.jsonl"
        if named.exists():
            return named
        sys.exit(f"no trace at {arg!r} or {named}")
    files = glob.glob(str(trace_dir() / "*.jsonl"))
    if not files:
        sys.exit(f"no traces in {trace_dir()}")
    return Path(max(files, key=os.path.getmtime))


def stage_of(span_name: str) -> str:
    return "tool" if span_name.startswith("tool.") else span_name


def turn_rows(root, events: list[ev.Event]) -> list[dict]:
    """One row per turn: what was said, what ran, and the stage bars.

    Turns come from the span tree so the timings match the dashboard exactly. The WORDS come
    from the raw events, because spans deliberately carry none — see core/otel.
    """
    turns = [s for s in root.children if s.name == "turn"]
    rows = []
    for i, turn in enumerate(turns):
        # Everything up to the next turn, so a reply and its tool calls stay with the turn that
        # caused them even when they land after the caller's next silence.
        end = turns[i + 1].start if i + 1 < len(turns) else float("inf")
        window = [e for e in events if turn.start <= e.t < end]

        said = ""
        confidence = None
        for e in window:
            if isinstance(e, ev.FinalTranscript) and not said:
                said, confidence = e.text, e.confidence

        replies: dict[str, str] = {}
        for e in window:
            if isinstance(e, ev.LLMTextDelta):
                replies[e.request_id] = replies.get(e.request_id, "") + e.text
            elif isinstance(e, ev.LLMCompleted) and e.text:
                replies.setdefault(e.request_id, e.text)

        rows.append({
            "index": i,
            "span": turn,
            "said": said,
            "confidence": confidence,
            "reply": " ".join(t.strip() for t in replies.values() if t.strip()),
            "tools": [(e.name, e.ok, e.latency_ms, e.http_status,
                       str(e.result.get("error", ""))[:120])
                      for e in window if isinstance(e, ev.ToolCompleted)],
            "routing": [(e.tier, e.model, e.reason) for e in window
                        if isinstance(e, ev.ModelRouted)],
            "intent": next((f"{e.intent} ({e.confidence:.2f})" for e in window
                            if isinstance(e, ev.IntentClassified)), ""),
            "held": [e.tail for e in window if isinstance(e, ev.TurnHeld) and e.released],
            # Kept off the waterfall (see waterfall_children) but reported, because "the agent
            # talked for 15 seconds" is a real finding — it is what a caller barges in on.
            "spoke_ms": sum(c.duration_ms for c in turn.children if c.name == "playback"),
        })
    return rows


def waterfall_children(turn) -> list:
    """The spans that make up a turn's LATENCY.

    `playback` is excluded and it is the whole reason this function exists. A turn ends at
    first audio out, so playback happens AFTER it — 15.6 s of it against a 2.5 s turn on one
    corpus call. Drawn to the same scale it is six times the chart and the stages the chart
    exists to show are slivers; drawn clipped it is a lie. Its duration goes on the meta line
    instead, which is where "the bot talked for 15 seconds" is actually useful.
    """
    return sorted((c for c in turn.children if c.name != "playback"), key=lambda s: s.start)


def bars(turn, scale_ms: float) -> str:
    """The waterfall for one turn: each child span as a bar, offset by when it started."""
    out = []
    for child in waterfall_children(turn):
        offset = (child.start - turn.start) * 1000
        width = max(child.duration_ms, 2.0)
        label = child.name.replace("tool.", "")
        color = STAGE_COLORS.get(stage_of(child.name), "#777")
        out.append(
            f'<div class="bar" style="margin-left:{offset / scale_ms * 100:.2f}%;'
            f'width:{width / scale_ms * 100:.2f}%;background:{color}" '
            f'title="{html.escape(label)} — {child.duration_ms:.0f} ms">'
            f'<span>{html.escape(label)} {child.duration_ms:.0f}ms</span></div>'
        )
    return "".join(out)


def render(path: Path) -> str:
    events = load_trace(path)
    root = spans_from_events(events)
    if root is None:
        sys.exit(f"{path.name} has no events")
    violations = tier1.check_events(events)
    rows = turn_rows(root, events)

    a = root.attributes
    answered = [r for r in rows if r["span"].attributes.get("turn.answered")
                and not r["span"].attributes.get("turn.held")]
    durations = sorted(r["span"].duration_ms for r in answered)
    p50 = durations[len(durations) // 2] if durations else None
    worst = max(answered, key=lambda r: r["span"].duration_ms, default=None)
    # The scale every waterfall shares. One scale across the whole call is the point: a slow
    # turn has to LOOK slow next to a fast one, which per-row scaling would hide. Taken from
    # where the bars actually END, not from the turn duration — a tool that returns just after
    # first audio legitimately overruns its turn, and scaling to the turn would push it off the
    # right edge.
    scale = max(
        [r["span"].duration_ms for r in rows]
        + [(c.start - r["span"].start) * 1000 + c.duration_ms
           for r in rows for c in waterfall_children(r["span"])],
        default=1000.0,
    ) or 1000.0

    tools_ran = [t for r in rows for t in r["tools"]]
    verdict = []
    if not tools_ran:
        verdict.append(("bad", "NO TOOL CALLS AT ALL — nothing was booked, changed, or looked up."))
    if a.get("call.booked"):
        verdict.append(("good", "A booking was committed: confirm_booking returned ok."))
    elif any(t[0] == "confirm_booking" for t in tools_ran):
        verdict.append(("bad", "confirm_booking ran and did NOT succeed."))
    for v in violations:
        verdict.append(("bad", f"Tier-1 invariant violated — {v}"))
    if not verdict:
        verdict.append(("ok", "No Tier-1 invariant was violated on this call."))

    def esc(text) -> str:
        return html.escape(str(text))

    parts = [f"""<!doctype html>
<meta charset="utf-8">
<title>{esc(a.get('call.id', path.stem))} — call trace</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#14161a; color:#dfe3e8;
         font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  main {{ max-width:1180px; margin:0 auto; padding:28px 20px 80px; }}
  h1 {{ font-size:19px; margin:0 0 4px; font-weight:600; }}
  .sub {{ color:#8b939e; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
           gap:10px; margin-bottom:22px; }}
  .stat {{ background:#1b1e24; border:1px solid #262b33; border-radius:6px; padding:10px 12px; }}
  .stat b {{ display:block; font-size:19px; font-weight:600; }}
  .stat span {{ color:#8b939e; font-size:12px; }}
  .verdict {{ border-radius:6px; padding:10px 14px; margin-bottom:8px; border-left:3px solid; }}
  .bad {{ background:#2a1618; border-color:#cf5f5f; color:#f0b6b6; }}
  .good {{ background:#15251b; border-color:#4aa06b; color:#a9dcbc; }}
  .ok {{ background:#1b1e24; border-color:#3a4049; color:#9aa3ae; }}
  .turn {{ border-top:1px solid #262b33; padding:14px 0; }}
  .turn.slow {{ background:#1a1518; }}
  .meta {{ color:#6f7883; font-size:12px; }}
  .caller {{ color:#7fc4ee; }} .agent {{ color:#dfe3e8; }}
  .quote {{ margin:4px 0 4px 1.5em; }}
  .waterfall {{ margin:8px 0 4px; }}
  .bar {{ height:17px; border-radius:2px; overflow:hidden; margin-bottom:2px;
          white-space:nowrap; min-width:2px; }}
  .bar span {{ font-size:10px; color:#10131a; padding-left:5px; line-height:17px; }}
  .tool {{ margin-left:1.5em; }}
  .fail {{ color:#e88; }}
  .legend span {{ margin-right:14px; font-size:12px; color:#8b939e; }}
  .swatch {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }}
  footer {{ margin-top:30px; color:#6f7883; font-size:12px; border-top:1px solid #262b33;
            padding-top:14px; }}
</style>
<main>
<h1>{esc(a.get('call.id', path.stem))}</h1>
<div class="sub">{esc(path)} · {len(events)} events · {a.get('call.mode', '?')}</div>
<div class="grid">
  <div class="stat"><b>{root.duration_ms / 1000:.0f}s</b><span>call duration</span></div>
  <div class="stat"><b>{a.get('call.turns', 0)}</b><span>turns</span></div>
  <div class="stat"><b>{f'{p50:.0f}ms' if p50 else '—'}</b><span>p50 voice-to-voice</span></div>
  <div class="stat"><b>{len(tools_ran)}</b><span>tool calls</span></div>
  <div class="stat"><b>{esc(a.get('call.intent', '—'))}</b><span>intent</span></div>
  <div class="stat"><b>{'yes' if a.get('call.booked') else 'no'}</b><span>booking committed</span></div>
</div>"""]

    for kind, text in verdict:
        parts.append(f'<div class="verdict {kind}">{esc(text)}</div>')

    parts.append('<p class="legend" style="margin-top:20px">' + "".join(
        f'<span><i class="swatch" style="background:{c}"></i>{esc(n)}</span>'
        for n, c in STAGE_COLORS.items()
    ) + f'<span>bars share one scale: {scale:.0f} ms full width</span></p>')

    for row in rows:
        turn = row["span"]
        held = turn.attributes.get("turn.held")
        unanswered = not turn.attributes.get("turn.answered")
        # "Slow" is relative to this call's own p50, not an absolute threshold: what is worth
        # looking at is the turn that stands out from its neighbours.
        slow = p50 is not None and not held and not unanswered and turn.duration_ms > p50 * 1.6
        note = " · HELD (caller paused)" if held else (" · no reply" if unanswered else "")
        parts.append(f'<div class="turn{" slow" if slow else ""}">')
        parts.append(
            f'<div class="meta">turn {row["index"]} · {turn.start - root.start:.1f}s · '
            f'{turn.duration_ms:.0f} ms{esc(note)}'
            + (f' · ASR {row["confidence"]:.2f}' if row["confidence"] is not None else "")
            + (f' · intent {esc(row["intent"])}' if row["intent"] else "")
            + "".join(f' · {esc(t)}/{esc(m)}' for t, m, _ in row["routing"])
            + (f' · spoke {row["spoke_ms"] / 1000:.1f}s' if row["spoke_ms"] else "")
            + "</div>"
        )
        if row["said"]:
            parts.append(f'<div class="quote caller">CALLER: {esc(row["said"])}</div>')
        for tail in row["held"]:
            parts.append(f'<div class="quote meta">(talked over mid-sentence: {esc(tail)})</div>')
        if row["reply"]:
            parts.append(f'<div class="quote agent">AGENT: {esc(row["reply"])}</div>')
        for name, ok, latency, status, error in row["tools"]:
            mark = "ok" if ok else "FAILED"
            detail = f" — {esc(error)}" if error else ""
            parts.append(
                f'<div class="tool{"" if ok else " fail"}">{esc(name)} · {mark} · '
                f'{latency:.0f} ms · HTTP {status or "—"}{detail}</div>'
            )
        parts.append(f'<div class="waterfall">{bars(turn, scale)}</div>')
        parts.append("</div>")

    parts.append(
        '<footer>Waterfall and turn boundaries come from <code>core/otel.spans_from_events</code>'
        ' — the same span tree the Grafana dashboard uses, so the two cannot disagree. The'
        ' verdict comes from <code>eval/tier1_replay</code>, the invariants CI runs on every'
        ' commit.<br>This page contains what the caller said. It lives in git-ignored'
        ' <code>logs/</code> and is not for sharing; the redacted view of the same call is the'
        ' OTel trace.'
        + (f'<br>Slowest answered turn: {worst["span"].duration_ms:.0f} ms at turn '
           f'{worst["index"]}.' if worst else "")
        + "</footer></main>"
    )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", nargs="?", help="path or call id (default: newest)")
    parser.add_argument("-o", "--out", type=Path, help="output file (default: logs/viewer/<id>.html)")
    parser.add_argument("--open", action="store_true", help="open it in a browser")
    args = parser.parse_args()

    path = resolve(args.trace)
    out = args.out or (trace_dir().parent / "viewer" / f"{path.stem}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(path), encoding="utf-8")

    print(f"{path.name} -> {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
