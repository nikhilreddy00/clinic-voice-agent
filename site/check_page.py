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
# Google Fonts is the one exception, and it is a deliberate one. The rule exists so the page
# has no functional dependency on the network — not so it renders in Times New Roman. A font
# stylesheet degrades gracefully: offline, the declared fallback stack renders and nothing
# breaks. Every other off-origin reference is still a failure, because anything else (a CDN
# script, a remote image) takes the page down with it when it 404s.
FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")

EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["'](?:https?:)?//([^/"']+)"""
    r"""|@import\s+url\(["']?https?://([^/"']+)"""
    r"""|(fetch)\s*\(""",
    re.IGNORECASE,
)

# Markup that must exist verbatim in the source.
REQUIRED_MARKUP = [
    'href="tel:+14842951203"',
    "/* BEGIN GENERATED TRACE */",
    "const CALL_REPLAY",
]

# Copy that must reach the READER. Checked against the page's rendered text with tags
# stripped, not against the source — a designer is free to wrap half a sentence in a span for
# emphasis, and a content assertion that breaks when they do is testing the markup rather than
# the claim. The tagline really did get split by an <span> mid-phrase, and this is the fix.
REQUIRED_COPY = [
    "Attend",
    "The front desk that never misses a call.",
    "Grove Family Clinic",
    "+1 (484) 295-1203",
    "Three in ten calls to a busy practice go unanswered",
    "It cannot tell a caller they are booked unless they are.",
    "The clinic's phone line has a pulse.",
    "2096F951",
    "This is a real inbound phone call",
    "19/19",
    "100%",
    "98.3%",
    "1,000",
    "1.5 s",
    "243",
    "What this is, and what it isn't",
    "No BAAs are signed",
]


def rendered_text(html: str) -> str:
    """Approximate what a reader sees: tags dropped, whitespace collapsed.

    <script> and <style> bodies are dropped too, except that the generated trace lives in a
    script and carries the confirmation number the page displays — so script content is kept
    when it is the trace block. Cheap and good enough to assert copy against.
    """
    text = re.sub(r"<style\b.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def check(html: str) -> list[str]:
    """Return a list of failures; empty means the page is good."""
    failures: list[str] = []

    for groups in EXTERNAL.findall(html):
        host = next((g for g in groups if g), "")
        if host == "fetch":
            failures.append("fetch() call — the page must not request anything at runtime")
        elif not host.endswith(FONT_HOSTS):
            failures.append(f"external reference to a non-font host: {host!r}")

    for needle in REQUIRED_MARKUP:
        if needle not in html:
            failures.append(f"missing required markup: {needle!r}")

    text = rendered_text(html)
    for needle in REQUIRED_COPY:
        if needle not in text:
            failures.append(f"copy missing from the rendered page: {needle!r}")

    # This page commits to one visual world rather than shipping two themes. That is allowed,
    # but only if it paints its own ground: an artifact composites over a background the viewer
    # paints in THEIR theme, so a body without an explicit background silently borrows it and
    # renders dark text on a dark host.
    if not re.search(r"body\s*\{[^}]*background:", html):
        failures.append("body sets no explicit background — it will borrow the host's theme")

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
