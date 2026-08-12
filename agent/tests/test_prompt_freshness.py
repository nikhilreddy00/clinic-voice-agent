"""Guards the date-freshness fix in pipeline.py's _reset_context_for_call().

The Phase-2 system prompt embeds today's clinic-local date and a 14-day date->weekday table.
The telephony worker is long-lived (it idles for days waiting for inbound calls), so a prompt
built once at process start silently goes stale at the first midnight and the model resolves
"tomorrow" against the wrong table.

Two things have to keep holding for the fix to work, and neither is obvious from reading
pipeline.py alone:

  1. The prompt actually varies with the wall clock (otherwise rebuilding it is pointless).
  2. LLMContext still exposes set_messages() as the way to replace the conversation -- `messages`
     is a read-only property, so this is the only supported seam. A pipecat change here would
     break the fix silently, which is why this is a contract test against real pipecat rather
     than a mock (same reasoning as test_metrics.py's VAD-frame canary).
"""

from datetime import datetime, timedelta, timezone

from pipecat.processors.aggregators.llm_context import LLMContext

from clinic_agent.prompts import build_phase2_system_prompt


def _system_text(context: LLMContext) -> str:
    return context.messages[0]["content"]


def test_prompt_date_table_changes_across_midnight():
    """A prompt built 'yesterday' must not equal one built 'today' -- else the fix is a no-op."""
    day1 = datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)

    assert build_phase2_system_prompt(now=day1) != build_phase2_system_prompt(now=day2)


def test_prompt_states_the_correct_weekday_for_today():
    """The injected table is the model's ground truth for date arithmetic -- verify one row."""
    # 2026-08-12 15:00 UTC is 11:00 EDT the same day, so clinic-local "today" is 2026-08-12.
    prompt = build_phase2_system_prompt(now=datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc))

    assert "2026-08-12 = Wednesday (today)" in prompt
    assert "2026-08-13 = Thursday (tomorrow)" in prompt


def test_llm_context_set_messages_replaces_conversation():
    """Contract test: set_messages() is the seam _reset_context_for_call() depends on.

    It must (a) exist, and (b) REPLACE rather than append -- replacement is what drops a prior
    caller's turns when one telephony process serves a second call.
    """
    context = LLMContext(messages=[{"role": "system", "content": "stale prompt"}])
    context.add_message({"role": "user", "content": "previous caller's turn"})
    assert len(context.messages) == 2

    context.set_messages([{"role": "system", "content": "fresh prompt"}])

    assert len(context.messages) == 1, "set_messages must replace, not append"
    assert _system_text(context) == "fresh prompt"


def test_rebuilding_context_yields_a_fresh_date_table():
    """End-to-end shape of the fix: stale prompt in, current prompt out, history cleared."""
    stale = build_phase2_system_prompt(now=datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc))
    context = LLMContext(messages=[{"role": "system", "content": stale}])
    context.add_message({"role": "user", "content": "book me for tomorrow"})

    # What _reset_context_for_call() does at the start of each call.
    context.set_messages([{"role": "system", "content": build_phase2_system_prompt()}])

    assert _system_text(context) != stale
    assert len(context.messages) == 1
