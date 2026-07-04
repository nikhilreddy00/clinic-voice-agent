"""Prompt text for the clinic voice agent.

Phase 0: placeholders only. The AI-disclosure line is mandatory (governance) and must appear
in the greeting whenever the agent runs. Detailed dialogue-state prompting is authored in
Phase 1/3 following docs/build_spec.md.
"""

# Mandatory AI disclosure. Delivered in the GREETING_DISCLOSURE state.
AI_DISCLOSURE = (
    "You're speaking with an automated AI assistant."
)

# Call-recording consent. Added to the greeting once telephony lands (Phase 7).
# TODO(Phase 7): confirm exact consent wording for the target jurisdiction.
RECORDING_CONSENT = (
    "This call may be recorded for quality and scheduling purposes."
)

# Greeting delivered at the start of the call (GREETING_DISCLOSURE).
GREETING = (
    "Thanks for calling Grove Family Clinic. "
    f"{AI_DISCLOSURE} "
    "I can help you book an appointment. How can I help today?"
)

# System prompt placeholder. Phase 1/3 will expand this into a full instruction set that
# encodes the state machine in docs/build_spec.md (intent -> name -> reason -> offer -> confirm
# -> book -> close), tool-calling rules for the scheduling API, and PHI-minimization guidance.
SYSTEM_PROMPT = """\
You are the virtual scheduling assistant for Grove Family Clinic, speaking with a caller by
phone. Keep replies short and natural for speech.

TODO(Phase 1/3): flesh out the full system prompt, including:
  - the dialogue state machine (see docs/build_spec.md)
  - tool-calling instructions for the scheduling API (availability / hold / confirm)
  - PHI minimization: collect only a coarse reason for the visit; never solicit clinical detail
  - fallback / no-match handling and escalation to a human
All patient data in development is synthetic.
"""
