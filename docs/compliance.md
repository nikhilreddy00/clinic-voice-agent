# BAA readiness — gap analysis

**This system is not HIPAA compliant and does not claim to be. All data in it is synthetic.**

What it is: an inbound scheduling voice agent built so that becoming compliant is *configuration
plus signed contracts* rather than a rewrite. This document is the honest version of that claim —
what is actually enforced in code today, what a signed BAA would change, what it would cost, and
what remains open. A clinic's IT reviewer wants this document, not a badge.

Last updated: 2026-09-07 (Phase 17). Every claim below is backed by a named test.

---

## 1. The contradiction, stated plainly

Legally taking real patient calls means a BAA with **every** vendor that touches the audio or the
text derived from it — five of them here: telephony, STT, LLM, TTS, and the hosting platform.
Each is an enterprise contract, and Anthropic's HIPAA-eligible configuration additionally cannot
run with zero-data-retention enabled. That is a four-figure-per-month floor before the first
call.

This project runs on ~$20–50/month of consumption pricing. **At that budget you cannot legally
handle PHI**, and no amount of engineering changes it. So the engineering target was the other
half of the problem: make the *architecture* the part that is already done.

## 2. Vendor matrix

| Layer | Vendor here | BAA available | What signing changes |
|---|---|---|---|
| Telephony / SIP | LiveKit Cloud | Yes, on enterprise plans | Call audio and SIP metadata become covered; free tier is not |
| STT | Deepgram | Yes, enterprise | Audio and transcripts covered; retention configurable |
| LLM | Anthropic (Claude Haiku 4.5) | Yes, commercial with BAA | **ZDR and HIPAA are mutually exclusive** — the covered configuration retains data under contract rather than not retaining it |
| TTS | Cartesia | Enterprise, case by case | Only the agent's own speech leaves; a caller's words reach TTS only if the model repeats them (it does — names, times) |
| Platform / DB | Supabase | Yes, on Team plan and above | The free tier is explicitly not covered, and it pauses after ~7 days idle |

Two consequences worth naming because they are counter-intuitive:

* **ZDR is not the compliant setting.** It is the setting you want when you are *not* covered.
  Under a BAA the vendor retains data as a business associate, with the contractual obligations
  that go with it.
* **TTS is in scope.** It is tempting to think only STT and the LLM see PHI. The agent speaks
  the caller's name and appointment back to them, so the TTS vendor receives PHI too.

## 3. What is enforced in code today

Each row is a property with a test that fails if it regresses. Two of them — the PHI log line
and the tenant filter — are planted as deliberate regressions by `scripts/prove_ci_gate.sh`
(five scenarios in total), which asserts the suite goes red on each.

| Control | Where | Test |
|---|---|---|
| One tagged PHI boundary for logs | `agent/src/clinic_agent/phi.py` | `agent/tests/test_phi.py` — every tool driven with sentinel values through the real logger |
| Closed allowlist for exported spans | `core/otel.ATTRIBUTES` (`_attrs` raises on anything else) | `agent/tests/test_otel.py` |
| Operational metrics cannot carry clinical fields | `scheduling_api/app/models.py` (`extra="forbid"`) | `agent/tests/test_call_metrics_shipping.py` |
| Date of birth is not stored in the clear | `db.dob_key` — HMAC-SHA256 under `CLINIC_PHI_KEY` | `scheduling_api/tests/test_phi_at_rest.py` |
| Clinical notes encrypted at rest | pgcrypto `pgp_sym_encrypt` in `db.confirm_booking` | same |
| Every PHI access is logged, denials included | `db._audited_verify` | `scheduling_api/tests/test_audit.py` |
| The audit log cannot be rewritten | trigger on `audit_log` (UPDATE/DELETE raise) | same |
| Tenants cannot read each other's patients | `clinic_id` on every query in `db.py` | `scheduling_api/tests/test_tenancy.py` |
| A phone number is not authentication | `db._verify`, two factors, fails closed | `scheduling_api/tests/test_verified_flows.py` |
| Wrong DOB and unknown number are indistinguishable | one 403 body, `main._NOT_VERIFIED` | same |
| A partial date is not a credential | `db.is_full_dob` + `reducer._is_full_dob` | `agent/tests/test_verification.py` (contract test imports both) |
| RLS deny-by-default on every table | `schema.sql` | `scheduling_api/tests/test_rls.py` |
| TLS to the database | `sslmode` on the Supabase URL | — (deployment configuration) |

### The two switches an operator sets

```bash
CLINIC_PHI_KEY=<32+ random bytes>   # DOB digest + note encryption. Unset = plaintext, and the
                                    # API says so in a warning on every boot.
CLINIC_PHI_LOGS=0                   # Console transcripts and trace-file transcripts off.
                                    # Structure, timings and tool calls are kept; the words go.
```

Both default to the *un*protected setting, deliberately and visibly, because this build runs on
synthetic data and the entire live-debugging workflow (`inspect_call.py`, `trace_viewer.py`, the
Tier-1 replay corpus) is built on full traces. **Neither default is safe for real PHI, and both
are one environment variable.**

### The key is part of the data

`CLINIC_PHI_KEY` digests the date of birth one-way. Change it or lose it and every enrolled
patient becomes unverifiable — there is no recovery path, by design, because that is what
one-way means. Rotating it is a re-enrollment, not a migration.
`test_changing_the_key_makes_every_caller_unverifiable` pins that it fails **closed**.

## 4. What is still open

The section that makes the rest of this document worth reading.

1. **`patient_name` is stored in the clear.** It is spoken back to the caller and fuzzy-matched
   during verification, and the application holds the encryption key next to the database, so
   encrypting it defends against a stolen disk and nothing else. A deliberate choice, not an
   oversight — but it is a gap, and a reviewer should count it as one.
2. **Transcripts are on disk by default.** `logs/traces/*.jsonl` holds the caller's words. Set
   `CLINIC_PHI_LOGS=0` for any real deployment.
3. **The application holds its own key.** `CLINIC_PHI_KEY` sits in the process environment.
   Real key management (KMS, envelope encryption, rotation) is not built.
4. **No retention or deletion policy.** Nothing ages out: not bookings, not audit rows, not
   traces. HIPAA does not mandate a retention period for this data, but a covered entity's own
   policy will, and there is no mechanism here to implement one.
5. **Authentication is one shared service token, not per tenant.** `X-Clinic-Slug` selects a
   tenant; it does not prove entitlement to it. The isolation behind the header is real and
   tested; the authentication in front of it is not per-tenant. A hostile client holding the
   token could name any clinic.
6. **The session router's LiveKit webhook is unauthenticated.** Verify the signing JWT before
   exposing it (Phase 11 note, still open).
7. **No BAA is signed with any vendor**, and the free tiers in use are explicitly excluded from
   the vendors' covered offerings.
8. **No formal risk analysis, workforce training, or incident-response plan.** These are
   organisational requirements of the Security Rule and are not engineering artifacts.
9. **Audit rows record access, not disclosure.** They say a chart was reached and by which call;
   they do not record what was read back to the caller. Reconstructing that needs the trace,
   which is the file item 2 wants turned off.

## 5. What "BAA-ready" means here

The claim is narrow and testable: **every place PHI crosses a boundary is one named, tested
place**, so making this system compliant is contracts and configuration rather than a redesign.

* PHI in **logs and spans** → `phi.py` and `otel.ATTRIBUTES`, both deny-by-default.
* PHI **at rest** → two columns, one key, one boot migration.
* PHI **access** → one audit helper, on the one verification path every PHI endpoint shares.
* PHI **per tenant** → `clinic_id` on every query, proven by a test that seeds the same phone
  number and birthday at two clinics.
* PHI **per vendor** → each provider sits behind an adapter (`core/adapters/`), so a swap to a
  HIPAA-eligible endpoint is a constructor argument, not a rewrite.

What it does **not** mean: that turning the switches on makes this compliant. It does not. Items
1–9 above are open, and five vendor contracts do not exist.
