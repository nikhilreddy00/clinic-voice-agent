"""Environment / settings loading for the clinic voice agent.

Phase 0: light boilerplate only. Values are read from a local (git-ignored) `.env`.
No secrets are hardcoded and nothing here makes network calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # loads .env from the agent/ working dir if present

# Fixed LiveKit room the inbound SIP call is routed into (Phase 5, Direct dispatch). The
# agent joins THIS room and the SIP dispatch rule points every inbound call at it, so the
# name must match in both places — importing this constant from both the agent pipeline and
# scripts/setup_livekit_sip.py is what keeps them in lockstep. (Phase 6 concurrency would
# switch to per-call rooms via an Individual dispatch rule + agent dispatch; see the script.)
TELEPHONY_ROOM_NAME = "clinic-inbound"


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration, populated from environment variables."""

    # Runtime mode: "local" = laptop mic/speaker (Phase 1), "telephony" = LiveKit SIP
    # inbound calls (Phase 5). Defaults to "local" so existing local testing is unchanged.
    mode: str

    # ASR — Deepgram
    deepgram_api_key: str

    # LLM — Anthropic (Claude), the active provider.
    anthropic_api_key: str
    anthropic_model: str

    # LLM — Groq (Llama). DORMANT fallback: kept for a future latency benchmark
    # only. Not wired into the active pipeline; switching back to Groq is a
    # deliberate code change in pipeline.py + eval/run_eval.py (see CLAUDE.md).
    groq_api_key: str
    groq_model: str

    # TTS — Cartesia (primary). Piper free-tier support is a later-phase TODO.
    cartesia_api_key: str
    cartesia_voice_id: str

    # Telephony — LiveKit SIP (required from Phase 5 onward)
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    livekit_phone_number: str

    # Scheduling backend (local mock API)
    scheduling_api_base_url: str


def load_settings() -> Settings:
    """Build a Settings object from the current environment.

    Phase 1 complete — required-key validation is enforced by require_phase1_keys()
    (and require_telephony_keys() for MODE=telephony), called at the pipeline entrypoint
    so a missing .env value fails fast with an actionable message.
    """
    return Settings(
        mode=os.getenv("MODE", "local").strip().lower(),
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", ""),
        # Telephony — LiveKit SIP. Not required until Phase 5, so these default
        # to "" and are not enforced by require_phase1_keys().
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        livekit_phone_number=os.getenv("LIVEKIT_PHONE_NUMBER", ""),
        scheduling_api_base_url=os.getenv(
            "SCHEDULING_API_BASE_URL", "http://127.0.0.1:8000"
        ),
    )


# Env vars the Phase-1 local voice loop (ASR -> LLM -> TTS) cannot run without.
# The active LLM is Anthropic/Claude, so ANTHROPIC_API_KEY is required here; a
# missing GROQ_API_KEY is fine — Groq is a dormant fallback (see CLAUDE.md).
_PHASE1_REQUIRED = {
    "DEEPGRAM_API_KEY": "deepgram_api_key",      # ASR
    "ANTHROPIC_API_KEY": "anthropic_api_key",    # LLM (active provider)
    "CARTESIA_API_KEY": "cartesia_api_key",      # TTS
    "CARTESIA_VOICE_ID": "cartesia_voice_id",    # TTS voice
}


def require_phase1_keys(settings: Settings) -> None:
    """Fail fast with a clear message if any key the Phase-1 pipeline needs is unset.

    Called at the top of the pipeline entrypoint so a missing `.env` value produces an
    actionable error instead of an opaque SDK auth failure mid-call.
    """
    missing = [
        env_name
        for env_name, field in _PHASE1_REQUIRED.items()
        if not getattr(settings, field)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s) for the Phase-1 voice loop: "
            f"{', '.join(missing)}. Set them in agent/.env."
        )


# Env vars the telephony path (MODE=telephony) needs on TOP of the Phase-1 keys: the
# LiveKit room URL + API credentials the agent uses to mint a join token and connect to
# the room the inbound SIP call is routed into. The phone number itself is only needed by
# the one-off dispatch-rule setup script, not by the running agent, so it is not required
# here.
_TELEPHONY_REQUIRED = {
    "LIVEKIT_URL": "livekit_url",
    "LIVEKIT_API_KEY": "livekit_api_key",
    "LIVEKIT_API_SECRET": "livekit_api_secret",
}


def require_telephony_keys(settings: Settings) -> None:
    """Fail fast if a key the LiveKit SIP path needs is unset (MODE=telephony only).

    Called before building the LiveKit transport so a missing credential produces an
    actionable error instead of an opaque connection failure once a real call arrives.
    """
    missing = [
        env_name
        for env_name, field in _TELEPHONY_REQUIRED.items()
        if not getattr(settings, field)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s) for MODE=telephony (LiveKit SIP): "
            f"{', '.join(missing)}. Set them in agent/.env."
        )
