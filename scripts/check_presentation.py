"""Verify presentation.md meets its structural constraint.

Every slide must carry exactly 8 bullet points, and every bullet must contain
exactly 9 whitespace-separated words. Counting fifty-six bullets by eye is a
reliable way to ship a mistake, so it is checked mechanically.

    python scripts/check_presentation.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

POINTS_PER_SLIDE = 8
WORDS_PER_POINT = 9

SLIDE = re.compile(r"^##\s+(Slide\s+\d+\s*[—-]\s*.+)$")
BULLET = re.compile(r"^-\s+(.*\S)\s*$")
STOP = re.compile(r"^###\s")


def slides(text: str) -> list[tuple[str, list[str]]]:
    """Collect (heading, bullets) pairs, stopping each slide at its first '###'."""
    found: list[tuple[str, list[str]]] = []
    heading: str | None = None
    bullets: list[str] = []
    collecting = False

    for line in text.splitlines():
        match = SLIDE.match(line)
        if match:
            if heading is not None:
                found.append((heading, bullets))
            heading, bullets, collecting = match.group(1).strip(), [], True
            continue
        if heading is None:
            continue
        if STOP.match(line) or line.startswith("## "):
            collecting = False
        bullet = BULLET.match(line)
        if bullet and collecting:
            bullets.append(bullet.group(1))

    if heading is not None:
        found.append((heading, bullets))
    return found


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "presentation.md"
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    parsed = slides(path.read_text(encoding="utf-8"))
    if not parsed:
        print("error: no slides found", file=sys.stderr)
        return 2

    failures = 0
    for heading, bullets in parsed:
        ok_count = len(bullets) == POINTS_PER_SLIDE
        marker = "OK  " if ok_count else "FAIL"
        print(f"\n{marker} {heading}")
        print(f"       points: {len(bullets)} (expected {POINTS_PER_SLIDE})")
        if not ok_count:
            failures += 1

        for index, bullet in enumerate(bullets, start=1):
            words = bullet.split()
            count = len(words)
            if count == WORDS_PER_POINT:
                print(f"       {index}. [{count}] {bullet}")
            else:
                failures += 1
                print(f"       {index}. [{count}] {bullet}   <-- expected {WORDS_PER_POINT}")

    print()
    total_points = sum(len(b) for _, b in parsed)
    print(f"slides: {len(parsed)}   points: {total_points}   failures: {failures}")
    if failures:
        print("CONSTRAINT NOT MET")
        return 1
    print("CONSTRAINT MET: every slide has 8 points, every point has 9 words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
