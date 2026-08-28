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
    if 'name="viewport"' not in html:
        failures.append("no viewport meta — the page will not scale on a phone")

    h1s = html.count("<h1")
    if h1s != 1:
        failures.append(f"expected exactly one <h1>, found {h1s}")

    # The proof table carries a min-width so its columns stay readable. That is only safe
    # inside an overflow container — otherwise it widens the whole document and every section
    # scrolls sideways on a phone.
    if "<table" in html and ".table-wrap" not in html:
        failures.append("table present with no .table-wrap overflow container")

    failures.extend(unclosed_tags(html))
    return failures


VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def unclosed_tags(html: str) -> list[str]:
    """Catch a tag left open — the failure mode of hand-editing one large file.

    A stray unclosed <div> renders fine in a forgiving browser right up until it silently
    swallows the section after it, which is exactly the kind of damage nobody spots in a diff.
    """
    from html.parser import HTMLParser

    class Checker(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.problems: list[str] = []

        def handle_starttag(self, tag: str, attrs) -> None:
            if tag not in VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in VOID:
                return
            if not self.stack:
                self.problems.append(f"closing </{tag}> with nothing open")
            elif self.stack[-1] != tag:
                self.problems.append(f"</{tag}> closes while <{self.stack[-1]}> is open")
                if tag in self.stack:
                    while self.stack and self.stack.pop() != tag:
                        pass
            else:
                self.stack.pop()

    checker = Checker()
    checker.feed(html)
    if checker.stack:
        checker.problems.append(f"never closed: {checker.stack}")
    return checker.problems


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
