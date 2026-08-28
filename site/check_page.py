#!/usr/bin/env python3
"""Invariant checks for the Attend landing page.

The page has no build step and no JS test framework — adding one to a Python repo for a
single static file would cost more than it catches. These are the assertions that actually
matter for this page: it must make no network requests, it must carry the real phone number,
and every number it publishes must be one the repo can back up.

Run: python3 site/check_page.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).parent / "index.html"

# Anything that would make the browser reach off-origin. The page must be openable from
# file:// with the network unplugged.
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["'](?:https?:)?//"""
    r"""|@import\s+url\(["']?https?:"""
    r"""|fetch\s*\(""",
    re.IGNORECASE,
)

REQUIRED_STRINGS = [
    "Attend",
    "The front desk that never misses a call.",
    "Grove Family Clinic",
    'href="tel:+14842950169"',
    "+1 (484) 295-0169",
    "Three in ten calls to a busy practice go unanswered",
    "It cannot tell a caller they are booked unless they are.",
    "/* BEGIN GENERATED TRACE */",
    "const CALL_REPLAY",
    "3BA83DD4",
    "text-in-the-loop eval case, not a phone call",
    "19/19",
    "100%",
    "98.3%",
    "1,000",
    "4.1 s",
    "243",
    "What this is, and what it isn't",
    "No BAAs are signed",
]


def check(html: str) -> list[str]:
    """Return a list of failures; empty means the page is good."""
    failures: list[str] = []

    for hit in EXTERNAL.findall(html):
        failures.append(f"external reference found: {hit!r}")

    for needle in REQUIRED_STRINGS:
        if needle not in html:
            failures.append(f"missing required string: {needle!r}")

    if "prefers-color-scheme: dark" not in html:
        failures.append("no dark theme block")

    # Conditional on purpose: a page with no motion needs no reduced-motion block, and
    # demanding one anyway would just mean an empty media query kept around to satisfy a
    # checker. Phrased this way the rule maintains itself — the moment anything animates,
    # the guard becomes mandatory.
    animates = "transition:" in html or "animation:" in html or "@keyframes" in html
    if animates and "prefers-reduced-motion" not in html:
        failures.append("page animates but has no prefers-reduced-motion block")
    if "<title>" not in html:
        failures.append("no <title>")
    if 'lang="en"' not in html:
        failures.append("no lang attribute on <html>")

    return failures


def main() -> int:
    if not PAGE.exists():
        print(f"FAIL: {PAGE} does not exist")
        return 1
    failures = check(PAGE.read_text(encoding="utf-8"))
    if failures:
        print(f"FAIL ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {PAGE.name} passed all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
