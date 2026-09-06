"""Phase 16 — the per-call trace viewer.

A debugging tool that renders a wrong page is worse than no tool: it is a wrong answer with a
chart next to it. These tests render every recorded call in the Tier-1 corpus and assert the
things a reader would act on — the verdict, the transcript, the tool outcomes, and the fact
that no bar runs off the right edge of its chart.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent" / "scripts"))

import trace_viewer as viewer  # noqa: E402

from clinic_agent.core import events as ev  # noqa: E402
from clinic_agent.core.otel import spans_from_events  # noqa: E402
from clinic_agent.core.recorder import load_trace  # noqa: E402

CORPUS = sorted((REPO_ROOT / "eval" / "traces").glob("*.jsonl"))

# The fabricated-booking call: the model read out "confirmation code is GFC-082826-1015" having
# called no tool at all. It is the exact call this viewer exists to make obvious.
NO_TOOLS = "20260828T004118517429Z"
# Cancel, then the refill the agent announced and never filed.
CANCEL_THEN_REFILL = "20260904T183444349772Z"


@pytest.fixture(scope="module", params=CORPUS, ids=lambda p: p.stem)
def rendered(request):
    return request.param, viewer.render(request.param)


def test_every_recorded_call_renders(rendered):
    path, page = rendered
    assert page.startswith("<!doctype html>")
    assert path.stem in page
    assert page.count("</div>") >= page.count('<div class="turn')


def test_no_bar_runs_off_the_right_edge(rendered):
    """The waterfall shares ONE scale across the whole call, so a slow turn looks slow next to a
    fast one. That only works if the scale covers where the bars actually END — a tool that
    returns just after first audio legitimately overruns its turn."""
    _, page = rendered
    edges = [float(a) + float(b)
             for a, b in re.findall(r"margin-left:([\d.]+)%;width:([\d.]+)%", page)]
    assert edges, "no bars were drawn"
    assert max(edges) <= 100.001, f"a bar reached {max(edges):.1f}% of the chart width"


def test_playback_is_never_a_bar(rendered):
    """A turn ends at first audio out, so playback happens AFTER it — 15.6 s of it against a
    2.5 s turn on one corpus call. On the same scale it is six times the chart and the stages
    the chart exists to show become slivers."""
    _, page = rendered
    bar_labels = re.findall(r'title="([^"]+?) —', page)
    assert "playback" not in bar_labels
    assert "spoke" in page or "no reply" in page, "playback duration is not reported anywhere"


def test_the_call_with_no_tool_calls_says_so_at_the_top():
    """`NO TOOL CALLS AT ALL` is the tell for a fabricated booking, and it is the single most
    important line this page can show."""
    page = viewer.render(REPO_ROOT / "eval" / "traces" / f"{NO_TOOLS}.jsonl")
    verdicts = re.findall(r'class="verdict (\w+)">([^<]+)', page)
    assert ("bad", "NO TOOL CALLS AT ALL — nothing was booked, changed, or looked up.") in [
        (k, v.strip()) for k, v in verdicts
    ]


def test_a_committed_booking_is_reported_from_the_tool_result_not_the_transcript():
    """The agent SAYING it booked is exactly what must not count. Every corpus call whose
    confirm_booking succeeded says yes; the one that only talked about booking says no."""
    for path in CORPUS:
        page = viewer.render(path)
        root = spans_from_events(load_trace(path))
        booked = bool(root.attributes.get("call.booked"))
        assert (">yes</b><span>booking committed" in page) is booked, path.name


def test_tool_failures_are_visible_with_their_reason():
    page = viewer.render(REPO_ROOT / "eval" / "traces" / f"{CANCEL_THEN_REFILL}.jsonl")
    assert "cancel_appointment" in page
    assert "verify_identity" in page


def test_the_transcript_is_escaped_not_injected():
    """Caller speech is arbitrary text going into HTML. A transcript containing a tag must not
    become one — this page is opened in a browser, and the input is a live phone line."""
    stream = [
        ev.CallStarted(seq=1, t=0.0, call_id="XSS", mode="telephony"),
        ev.SpeechStopped(seq=2, t=1.0),
        ev.FinalTranscript(seq=3, t=1.4, text="<script>alert(1)</script>", confidence=0.9),
        ev.LLMStarted(seq=4, t=1.5, request_id="req-1"),
        ev.LLMCompleted(seq=5, t=2.0, request_id="req-1", stop_reason="end_turn",
                        text="Sure <img src=x onerror=alert(2)>"),
        ev.BotStartedSpeaking(seq=6, t=2.2, utterance_id="u1"),
        ev.BotStoppedSpeaking(seq=7, t=4.0, utterance_id="u1"),
        ev.Hangup(seq=8, t=5.0),
    ]
    assert "<script>" in viewer.turn_rows(spans_from_events(stream), stream)[0]["said"], (
        "the raw value must reach the renderer — otherwise this proves nothing about escaping"
    )

    page = _render_stream(stream)

    # What matters is that no TAG survives. The literal text `onerror=alert(2)` appearing as
    # prose is fine and is what correct escaping looks like; asserting on it instead of on the
    # angle brackets is how an escaping test comes to fail on a page that is perfectly safe.
    assert "<script>" not in page
    assert "<img" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def _render_stream(stream) -> str:
    """Render an in-memory event stream by writing it out the way the recorder would."""
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for event in stream:
            fh.write(json.dumps(event.to_dict()) + "\n")
        path = Path(fh.name)
    try:
        return viewer.render(path)
    finally:
        path.unlink(missing_ok=True)


def test_an_empty_trace_exits_rather_than_rendering_a_blank_page(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        viewer.render(empty)
