# Tier-1 replay corpus

Real recorded calls (synthetic patient data — see CLAUDE.md), replayed by
`eval/tier1_replay.py` through the live reducer on every commit.

`logs/traces/` is git-ignored, so a trace has to be copied here to be a CI gate.

**Adding a trace.** Copy it from `logs/traces/`, run `eval/tier1_replay.py`, and check the
`[n/m tool calls replayed]` count. A trace whose count is `0/m` replays to nothing — request
ids renumber when the reducer changes, so an old recording's later events are dropped as stale
— and Tier 1 fails it rather than letting it pass vacuously. Re-record against the current
engine or leave it out.

**Removing a trace** is fine when it goes stale. Losing coverage is visible; a corpus of files
that replay to nothing is not.

| trace | what it covers |
|---|---|
| `20260828T004118517429Z` | greeting, AI disclosure, full intake slot-filling (no tools) |
| `20260829T182056478576Z` | booking end to end: availability → hold → confirm, read-back |
| `20260830T175851884586Z` | booking with a slot change mid-flow (6 tool calls) |
| `20260903T170114545113Z` | the double-book call — reschedule intent preempted by scheduling |
| `20260903T205341764653Z` | booking on a shared handset |
| `20260904T172643433047Z` | booking with new-patient intake and symptom notes |
| `20260904T183444349772Z` | cancel, then the refill the model announced and never filed |
