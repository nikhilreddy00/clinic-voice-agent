"""Environment / settings loading for the clinic voice agent.

Phase 0: light boilerplate only. Values are read from a local `.env` (see `.env.example`).
No secrets are hardcoded and nothing here makes network calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # loads .env from the agent/ working dir if present


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration, populated from environment variables."""

    # ASR — Deepgram
    deepgram_api_key: str

    # LLM — Groq (Llama)
    groq_api_key: str
    groq_model: str

    # TTS — Cartesia (primary). Piper free-tier support is a later-phase TODO.
    cartesia_api_key: str
    cartesia_voice_id: str

    # Telephony — Twilio (Phase 7)
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str

    # Scheduling backend (local mock API)
    scheduling_api_base_url: str


def load_settings() -> Settings:
    """Build a Settings object from the current environment.

    TODO(Phase 1): validate required keys and fail fast with a clear message when a key
    needed by the active pipeline is missing.
    """
    return Settings(
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", ""),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER", ""),
        scheduling_api_base_url=os.getenv(
            "SCHEDULING_API_BASE_URL", "http://127.0.0.1:8000"
        ),
    )


# Env vars the Phase-1 local voice loop (ASR -> LLM -> TTS) cannot run without.
_PHASE1_REQUIRED = {
    "DEEPGRAM_API_KEY": "deepgram_api_key",  # ASR
    "GROQ_API_KEY": "groq_api_key",          # LLM
    "CARTESIA_API_KEY": "cartesia_api_key",  # TTS
    "CARTESIA_VOICE_ID": "cartesia_voice_id",  # TTS voice
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
            f"{', '.join(missing)}. Set them in agent/.env (see .env.example)."
        )
