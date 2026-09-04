-- Phase-9 Postgres schema for the clinic scheduling service.
--
-- Replaces the Phase-0 SQLite schema. Three things drove the rewrite, in order of severity:
--
--   1. CORRECTNESS. The SQLite handlers read a slot's status and then wrote it in a separate
--      statement with no transaction and no guard, so two concurrent callers could both read
--      'available' and both write 'held' -- a double-booked slot. Every state transition here is
--      expressible as a single-statement compare-and-swap (see the queries in db.py), which is
--      atomic in Postgres without an explicit transaction or an advisory lock.
--   2. CONCURRENCY. SQLite has one writer. The /availability read path used to write on every
--      request (releasing expired holds and re-stamping slot dates), which serialised reads
--      behind that single writer lock. Those writes now live in a background sweeper.
--   3. TIMEZONE. Slot times were stored as naive UTC strings and compared lexically, which is
--      why seeded 9:00 slots were really 05:00 clinic-local. Times are timestamptz now, and
--      seed times are generated in the clinic's own zone.
--
-- Tables below the SCHEDULING section are landed now but unused until Phases 12-13 (intent
-- classification, memory, expanded tools). Creating them in this migration rather than a later
-- one keeps the production data layer to a single migration, and every table carries clinic_id
-- from the start -- retrofitting tenancy onto rows that already exist is far more painful than
-- carrying an unused column for a few weeks.
--
-- All data is synthetic. This schema is BAA-ready in shape (PHI is isolated to named columns,
-- every access is auditable) but is NOT a compliance claim -- see docs/compliance.md.

-- =========================================================================================
-- TENANCY
-- =========================================================================================

CREATE TABLE IF NOT EXISTS clinics (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug        TEXT        NOT NULL UNIQUE,
    name        TEXT        NOT NULL,
    timezone    TEXT        NOT NULL DEFAULT 'America/New_York',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================================================================================
-- SCHEDULING (in use today)
-- =========================================================================================

CREATE TABLE IF NOT EXISTS providers (
    id          BIGINT      PRIMARY KEY,
    clinic_id   BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    name        TEXT        NOT NULL,
    specialty   TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS providers_clinic_idx ON providers (clinic_id);

CREATE TABLE IF NOT EXISTS slots (
    id               BIGINT      PRIMARY KEY,
    clinic_id        BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    provider_id      BIGINT      NOT NULL REFERENCES providers(id),
    start_time       TIMESTAMPTZ NOT NULL,
    reason_category  TEXT        NOT NULL,
    -- A CHECK beats an enum here: adding a state later is an ALTER rather than a type
    -- migration, and the set is small and stable.
    status           TEXT        NOT NULL DEFAULT 'available'
                                 CHECK (status IN ('available', 'held', 'booked')),
    hold_id          UUID,
    hold_expires_at  TIMESTAMPTZ,

    -- A held slot must carry both hold columns; a slot in any other state must carry neither.
    -- Without this, a partial write (or a future bug) could leave a slot 'held' forever with no
    -- expiry, permanently un-bookable and invisible to the sweeper.
    CONSTRAINT slots_hold_fields_consistent CHECK (
        (status = 'held' AND hold_id IS NOT NULL AND hold_expires_at IS NOT NULL)
        OR (status <> 'held' AND hold_id IS NULL AND hold_expires_at IS NULL)
    )
);

-- The availability read path filters on (clinic, status, start_time) and orders by start_time.
CREATE INDEX IF NOT EXISTS slots_availability_idx
    ON slots (clinic_id, status, start_time)
    WHERE status = 'available';

-- confirm-booking looks a slot up by hold_id. Partial: only held slots have one.
CREATE UNIQUE INDEX IF NOT EXISTS slots_hold_id_idx
    ON slots (hold_id)
    WHERE hold_id IS NOT NULL;

-- The sweeper scans for expired holds; partial index keeps it off the whole table.
CREATE INDEX IF NOT EXISTS slots_expiring_holds_idx
    ON slots (hold_expires_at)
    WHERE status = 'held';

CREATE TABLE IF NOT EXISTS bookings (
    confirmation_id  TEXT        PRIMARY KEY,
    clinic_id        BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    -- The backstop against double-booking. The compare-and-swap in confirm_booking() should
    -- make this unreachable; it is here so that if a future refactor reintroduces the
    -- read-then-write race, the database refuses the second booking instead of accepting it.
    slot_id          BIGINT      NOT NULL UNIQUE REFERENCES slots(id),
    patient_id       BIGINT,  -- FK added with the patients table below; NULL until Phase 13
    patient_name     TEXT        NOT NULL,
    reason           TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    status           TEXT        NOT NULL DEFAULT 'confirmed'
                                 CHECK (status IN ('confirmed', 'cancelled')),
    -- PHI. Synthetic only. Isolated here so redaction/encryption has one place to act (Phase 17).
    date_of_birth    TEXT,
    new_patient      BOOLEAN,
    symptom_notes    TEXT
);

CREATE INDEX IF NOT EXISTS bookings_clinic_created_idx ON bookings (clinic_id, created_at DESC);

-- =========================================================================================
-- IDEMPOTENCY
-- =========================================================================================

-- A voice turn can time out after the write has already committed. Without this table the
-- agent's retry books a second appointment (or gets a spurious 409) because hold_id and
-- confirmation_id are freshly generated per request. The client sends a stable key; the first
-- request stores its response, and every retry replays that response rather than re-executing.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key           TEXT        NOT NULL,
    endpoint      TEXT        NOT NULL,
    -- Guards against a client reusing one key for two different payloads, which would otherwise
    -- silently return the first call's answer for the second call's request.
    request_hash  TEXT        NOT NULL,
    -- NULL while the request is in flight: the row is claimed before the work runs, so a
    -- concurrent duplicate can be detected rather than racing to do the same write twice.
    response_json JSONB,
    status_code   INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (key, endpoint)
);

CREATE INDEX IF NOT EXISTS idempotency_created_idx ON idempotency_keys (created_at);

-- =========================================================================================
-- PHASE 12-13: created now, unused until then (single-migration rule)
-- =========================================================================================

CREATE TABLE IF NOT EXISTS patients (
    id             BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id      BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    -- E.164. The lookup key for caller memory -- but recognising a number is NOT authentication;
    -- disclosure requires a verified DOB match (see docs, Phase 13).
    phone          TEXT,
    name           TEXT,
    date_of_birth  TEXT,       -- PHI
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Keyed on (phone, DOB), NOT on phone alone. A phone number is a household, not a person:
    -- spouses, children, and parents share one handset, and a clinic must be able to hold all
    -- of them. `UNIQUE (clinic_id, phone)` meant the FIRST caller from a number owned it
    -- forever — see db._upsert_patient and db._verify for the live failure that produced.
    --
    -- `date_of_birth` here is always the NORMALIZED form (db.normalize_dob), because a key
    -- made of free text would file "3/5/2001" and "03/05/2001" as two different people.
    UNIQUE (clinic_id, phone, date_of_birth)
);

CREATE TABLE IF NOT EXISTS caller_memory (
    id                    BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id             BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    patient_id            BIGINT      REFERENCES patients(id) ON DELETE CASCADE,
    phone                 TEXT,
    preferred_provider_id BIGINT      REFERENCES providers(id),
    preferences           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (clinic_id, phone)
);

CREATE TABLE IF NOT EXISTS call_summaries (
    id                BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id         BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    call_id           TEXT        NOT NULL,
    patient_id        BIGINT      REFERENCES patients(id) ON DELETE SET NULL,
    intent            TEXT,
    outcome           TEXT,
    escalation_reason TEXT,
    summary           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (clinic_id, call_id)
);

-- Per-tenant curated facts (hours, location, insurance, prep instructions), injected into the
-- prompt by intent. Deliberately a table rather than a vector store: at this volume a fact table
-- is faster, cheaper, and fully auditable.
CREATE TABLE IF NOT EXISTS clinic_facts (
    id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id  BIGINT NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    topic      TEXT   NOT NULL,
    content    TEXT   NOT NULL,

    UNIQUE (clinic_id, topic)
);

-- Work the agent must hand to a human rather than complete itself -- refill requests above all.
-- The agent creates the task; it never approves a refill.
CREATE TABLE IF NOT EXISTS staff_tasks (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id   BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    patient_id  BIGINT      REFERENCES patients(id) ON DELETE SET NULL,
    kind        TEXT        NOT NULL,
    payload     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status      TEXT        NOT NULL DEFAULT 'open'
                            CHECK (status IN ('open', 'done', 'cancelled')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS staff_tasks_open_idx ON staff_tasks (clinic_id, status, created_at);

-- Append-only. Every PHI read/write gets a row (Phase 17). No UPDATE/DELETE path by design.
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    clinic_id    BIGINT      NOT NULL REFERENCES clinics(id) ON DELETE CASCADE,
    actor        TEXT        NOT NULL,
    action       TEXT        NOT NULL,
    resource     TEXT        NOT NULL,
    resource_id  TEXT,
    call_id      TEXT,
    at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audit_log_clinic_at_idx ON audit_log (clinic_id, at DESC);

-- Deferred FK: bookings.patient_id -> patients.id. Declared here because bookings is created
-- before patients (bookings is in use today; patients is not).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'bookings_patient_id_fkey'
    ) THEN
        ALTER TABLE bookings
            ADD CONSTRAINT bookings_patient_id_fkey
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL;
    END IF;
END $$;

-- =========================================================================================
-- ROW LEVEL SECURITY
-- =========================================================================================
--
-- Deny-by-default on every table. This is a no-op for the service itself (it connects as a role
-- that bypasses RLS) and matters entirely for one deployment target: SUPABASE.
--
-- Supabase exposes the `public` schema through its Data API (PostgREST). Any table here is
-- therefore potentially reachable by the `anon` / `authenticated` roles with nothing more than
-- the project's publishable key -- which is, by design, shipped to browsers. `bookings` holds
-- patient_name, date_of_birth, and symptom_notes; `patients` and `call_summaries` are worse.
-- Without RLS that is a PHI disclosure reachable from a browser console.
--
-- RLS is enabled with NO policies, which denies everything to non-bypassing roles. That is the
-- correct posture here because this service is the only intended reader and it connects
-- directly as an owner/service role. Nothing should be querying these tables through the Data
-- API; if that ever changes, add explicit policies rather than disabling RLS.
--
-- Failure mode to know about: connect as a role WITHOUT bypassrls and every query returns zero
-- rows rather than an error. That is loud in the right way (obviously broken) instead of quiet
-- in the wrong way (data leaking).

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'clinics', 'providers', 'slots', 'bookings', 'idempotency_keys',
        'patients', 'caller_memory', 'call_summaries', 'clinic_facts',
        'staff_tasks', 'audit_log'
    ] LOOP
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = t) THEN
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
        END IF;
    END LOOP;
END $$;
