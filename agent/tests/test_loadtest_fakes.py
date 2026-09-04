"""The load test's fake adapters must keep the same shape as the real ones.

Phase 13 added `context_note` to `AnthropicLLM.start()` and a `load_caller_memory()` to the
tool executor. `loadtest/fake_adapters.py` was not updated, so from that commit until
2026-09-04 every Tier-A session raised on its first turn — and the harness printed a table of
`None`s, wrote a chart, and exited 0. A whole phase of load numbers measured nothing.

The load test's entire value is that it drives the REAL orchestrator, so a fake that has
drifted out of shape does not just fail, it quietly measures a different program. `tier_a.py`
now fails loudly when sessions die; this catches the drift before anyone runs it.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

LOADTEST = Path(__file__).resolve().parents[2] / "loadtest"
pytestmark = pytest.mark.skipif(not LOADTEST.is_dir(), reason="loadtest/ not present")
sys.path.insert(0, str(LOADTEST))

import fake_adapters as fakes  # noqa: E402

from clinic_agent.core.adapters.classifier import IntentClassifier  # noqa: E402
from clinic_agent.core.adapters.llm import AnthropicLLM  # noqa: E402
from clinic_agent.core.adapters.stt import DeepgramSTT  # noqa: E402
from clinic_agent.core.adapters.tools import ToolExecutor  # noqa: E402
from clinic_agent.core.adapters.tts import CartesiaTTS  # noqa: E402

PAIRS = [
    (fakes.FakeLLM, AnthropicLLM),
    (fakes.FakeClassifier, IntentClassifier),
    (fakes.FakeSTT, DeepgramSTT),
    (fakes.FakeTTS, CartesiaTTS),
    (fakes.FakeTools, ToolExecutor),
]


@pytest.mark.parametrize("fake,real", PAIRS, ids=lambda c: c.__name__)
def test_the_fake_adapter_still_matches_the_real_one(fake, real):
    """Every public method of the real adapter must exist on the fake, taking the same
    keywords. Extra keywords on the fake are fine; missing ones are the whole bug."""
    problems = []
    for name in dir(real):
        if name.startswith("_") or not callable(getattr(real, name, None)):
            continue
        if not hasattr(fake, name):
            problems.append(f"{fake.__name__} is missing {name}()")
            continue
        try:
            expected = inspect.signature(getattr(real, name)).parameters
            actual = inspect.signature(getattr(fake, name)).parameters
        except (ValueError, TypeError):
            continue  # C-level or otherwise un-introspectable; nothing to compare
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in actual.values()):
            continue  # **kwargs accepts anything
        missing = [k for k in expected if k not in actual and k != "self"]
        if missing:
            problems.append(f"{fake.__name__}.{name}() does not accept {missing}")
    assert not problems, (
        "loadtest/fake_adapters.py has drifted from the real adapters — Tier-A sessions will "
        "die on their first turn and the numbers will be meaningless:\n  "
        + "\n  ".join(problems)
    )
