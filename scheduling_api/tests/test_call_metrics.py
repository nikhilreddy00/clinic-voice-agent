"""Phase 16 — POST /call-metrics, and the round trip back out through /metrics.

The file bus this replaces was broken in the one deployment that matters (separate Railway
containers, no shared volume), unbounded, and re-read in full per request. These tests pin the
three properties the replacement has to have and the old one did not: it survives a retry, it
aggregates from raw turns rather than per-call summaries, and it refuses to carry PHI.
"""

from __future__ import annotations


def _payload(call_id: str = "CALL-1", **over) -> dict:
    body = {
        "call_id": call_id,
        "mode": "telephony",
        "outcome": "booked",
        "tool_total": 2,
        "tool_success": 1,
        "turns": [
            {"asr_ms": 200, "llm_ms": 400, "tts_ms": 300, "e2e_ms": 900,
             "asr_confidence": 0.9, "had_tool_call": False},
            {"asr_ms": 100, "llm_ms": 200, "tts_ms": None, "e2e_ms": 500,
             "asr_confidence": None, "had_tool_call": True},
        ],
        "tools": [
            {"endpoint": "/availability", "http_status": 200, "latency_ms": 30, "success": True},
            {"endpoint": "/hold-slot", "http_status": 409, "latency_ms": 12, "success": False},
        ],
    }
    body.update(over)
    return body


def test_a_posted_call_comes_back_out_of_metrics(client):
    resp = client.post("/call-metrics", json=_payload())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "call_id": "CALL-1", "turns": 2, "tools": 2}

    agg = client.get("/metrics").json()
    assert agg["calls_total"] == 1
    assert agg["turns_total"] == 2
    assert agg["latency_ms"]["e2e"]["count"] == 2
    assert agg["latency_ms"]["tts"]["count"] == 1     # the null turn contributes no sample
    assert agg["asr_confidence"]["count"] == 1
    assert agg["tool_calls"]["total"] == 2 and agg["tool_calls"]["success"] == 1
    assert agg["tool_calls"]["by_endpoint"]["/hold-slot"]["success"] == 0
    assert agg["outcomes"]["booked"] == 1
    assert agg["recent_calls"][0]["call_id"] == "CALL-1"


def test_reposting_a_call_replaces_it_rather_than_doubling_it(client):
    """The agent posts once at teardown, but "once" is a property of a network nobody owns.

    A duplicate would double-count in every percentile on the dashboard, which is the quiet kind
    of wrong: the page still renders, the numbers are just false.
    """
    client.post("/call-metrics", json=_payload())
    client.post("/call-metrics", json=_payload())

    agg = client.get("/metrics").json()
    assert agg["calls_total"] == 1
    assert agg["turns_total"] == 2
    assert agg["tool_calls"]["total"] == 2


def test_a_shorter_repost_does_not_leave_the_old_tail_behind(client):
    """Children are replaced wholesale, not upserted. An UPSERT keyed per row would keep turn 2
    of the first post alive forever."""
    client.post("/call-metrics", json=_payload())
    short = _payload()
    short["turns"] = short["turns"][:1]
    short["tools"] = []
    client.post("/call-metrics", json=short)

    agg = client.get("/metrics").json()
    assert agg["turns_total"] == 1
    assert agg["tool_calls"]["total"] == 0


def test_percentiles_span_calls(client):
    """Two calls' turns aggregate together — the whole reason turns are stored individually."""
    fast = _payload("FAST", tools=[], tool_total=0, tool_success=0,
                    turns=[{"e2e_ms": ms} for ms in (100, 110, 120)])
    slow = _payload("SLOW", tools=[], tool_total=0, tool_success=0,
                    turns=[{"e2e_ms": ms} for ms in (900, 1000, 4000)])
    client.post("/call-metrics", json=fast)
    client.post("/call-metrics", json=slow)

    agg = client.get("/metrics").json()
    assert agg["calls_total"] == 2
    assert agg["latency_ms"]["e2e"]["count"] == 6
    assert agg["latency_ms"]["e2e"]["p95"] == 4000


def test_metrics_is_empty_and_well_formed_before_any_call(client):
    agg = client.get("/metrics").json()
    assert agg["calls_total"] == 0
    assert agg["latency_ms"]["e2e"]["p50"] is None
    assert agg["recent_calls"] == []


# --- the PHI boundary, enforced by the schema rather than by discipline --------------------


def test_phi_fields_are_rejected_outright(client):
    """`extra="forbid"` on the request model is the enforcement.

    These rows have a retention policy that assumes no clinical content, and they sit in a
    different table from `call_summaries` for that reason. A future agent that attaches a name
    to a metric gets a 422, not a silent write into the wrong compartment. Phase 17 generalises
    this into one tagged boundary; this is the piece of it that is free today.
    """
    for field, value in (
        ("patient_name", "Nicholas Kumar"),
        ("date_of_birth", "12/08/2000"),
        ("transcript", "I need to book an appointment"),
        ("symptom_notes", "twisted ankle, pain 6/10"),
    ):
        resp = client.post("/call-metrics", json=_payload(**{field: value}))
        assert resp.status_code == 422, f"{field} was accepted onto an operational record"


def test_a_turn_cannot_smuggle_extra_fields_either(client):
    body = _payload()
    body["turns"][0]["utterance"] = "my date of birth is December 8th 2000"
    assert client.post("/call-metrics", json=body).status_code == 422


def test_a_blank_call_id_is_refused(client):
    assert client.post("/call-metrics", json=_payload(call_id="")).status_code == 422
