"""Screenshot the ChainGuard terminal for visual review.

Development tool, not part of the pipeline. Drives a headless browser over the
running console so the interface can be inspected at real breakpoints.

    python scripts/shoot.py --base http://127.0.0.1:8765 --out outputs/shots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIEWS = ["monitor", "market", "models", "drivers", "ledger", "system"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="outputs/shots")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--run-chain", action="store_true",
                        help="execute the agent chain before shooting the monitor view")
    parser.add_argument("--only", default=None, help="single view to capture")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = [args.only] if args.only else VIEWS
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=2)
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(args.base, wait_until="networkidle")
        page.wait_for_timeout(1200)

        if args.run_chain:
            page.fill("#f-date", "2022-11-05")
            page.click("#f-run")
            page.wait_for_timeout(6000)

        for view in views:
            page.click(f'.nav button[data-view="{view}"]')
            page.wait_for_timeout(2200)
            page.screenshot(path=str(out / f"{view}.png"))
            print(f"  {view:9s} -> {out / f'{view}.png'}")

        # Narrow breakpoint on the densest view.
        page.set_viewport_size({"width": 430, "height": 900})
        page.click('.nav button[data-view="monitor"]')
        page.wait_for_timeout(1200)
        page.screenshot(path=str(out / "narrow.png"), full_page=True)
        print(f"  narrow    -> {out / 'narrow.png'}")

        browser.close()

    if errors:
        print("\nBROWSER DIAGNOSTICS")
        for line in dict.fromkeys(errors):
            print("  " + line)
    else:
        print("\nno console errors or warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
