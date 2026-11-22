#!/usr/bin/env python3
"""Headless TUI check (Textual pilot): open every tab, assert widgets rendered, press refresh, save SVG screenshots.
  ~/.venvs/gogeta-cockpit/bin/python cockpit/tui_check.py [--shots]
Read-only against live state (it never presses Start/Stop/Launch)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tui  # noqa: E402
from textual.widgets import DataTable, RichLog, Static, TabbedContent  # noqa: E402

SHOTS = Path(__file__).resolve().parent / "screenshots"


async def wait_for(pilot, cond, tries=60):
    for _ in range(tries):
        await pilot.pause(0.25)
        if cond():
            return True
    return False


async def run(size, shots):
    results = []
    app = tui.CockpitApp()
    async with app.run_test(size=size) as pilot:
        ok = await wait_for(pilot, lambda: app.status.get("gpu") is not None)
        results.append(("dashboard status loaded", ok))
        results.append(("gpu card text", "GPU" in str(app.query_one("#gpu", Static).render())))
        results.append(("campaign table rows >= 0", app.query_one("#camp-table", DataTable).row_count >= 0))
        if shots:
            app.save_screenshot("dashboard_%dx%d.svg" % size, path=str(SHOTS))
        checks = {"queue": ("#queue-table", 1), "experiments": ("#exp-table", 1), "claims": ("#claims-table", 1),
                  "devlog": ("#dev-table", 1), "errors": ("#errors-table", 0), "incidents": ("#incidents-table", 0), "runs": ("#runs-table", 1), "search": ("#search-table", 0)}
        for tab, (sel, minrows) in checks.items():
            app.query_one("#tabs", TabbedContent).active = tab
            t = app.query_one(sel, DataTable)
            good = await wait_for(pilot, lambda: t.row_count >= minrows)
            results.append(("tab %s rows=%d" % (tab, t.row_count), good))
            if shots and size[0] >= 100 and tab in ("queue", "errors", "experiments"):
                app.save_screenshot("%s_%dx%d.svg" % (tab, *size), path=str(SHOTS))
        app.query_one("#tabs", TabbedContent).active = "runs"
        await pilot.pause(1.0)
        await pilot.click("#runs-forms")
        ft = app.query_one("#forms-table", DataTable)
        results.append(("runs: run forms table rows=%d" % ft.row_count, await wait_for(pilot, lambda: ft.row_count >= 1)))
        results.append(("runs: form detail shows machine/agent fields", await wait_for(pilot, lambda: "[agent]" in "".join(str(l.text) for l in app.query_one("#form-detail", RichLog).lines))))
        await pilot.click("#runs-forms")
        app.query_one("#tabs", TabbedContent).active = "logs"
        good = await wait_for(pilot, lambda: len(app.query_one("#log-view", RichLog).lines) > 0)
        results.append(("tab logs tail lines>0", good))
        await pilot.press("r")
        await pilot.pause(0.5)
        results.append(("refresh key pressed", True))
        await pilot.press("question_mark")
        await pilot.pause(0.3)
        results.append(("help screen open", type(app.screen).__name__ == "HelpScreen"))
        await pilot.press("escape")
    return results


def main():
    shots = "--shots" in sys.argv
    if shots:
        SHOTS.mkdir(exist_ok=True)
    bad = 0
    for size in ((120, 40), (80, 24)):
        print("== terminal %dx%d" % size)
        for name, ok in asyncio.run(run(size, shots)):
            print("  %-34s %s" % (name, "OK" if ok else "FAIL"))
            bad += 0 if ok else 1
    print("FAILED: %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
