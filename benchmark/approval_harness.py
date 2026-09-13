#!/usr/bin/env python3
"""Deterministic JupyterLab consent harness for benchmark trials.

The harness keeps the trial notebook active so the peaksMCP Comm bridge stays
connected. An empty expected-file allowlist denies every persistence dialog,
as required by the current Notebook-only benchmark. Every decision is appended
to an operator JSONL log.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAVE_TITLE = re.compile(r"peaksMCP.*(?:保存|save)", re.IGNORECASE)
SAVE_DIALOG_KIND = "save-consent"
_POSIX_PRODUCT = re.compile(r"/[^\n\r]*?\.nc")
_WINDOWS_PRODUCT = re.compile(r"[A-Za-z]:\\[^\n\r]*?\.nc")


def extract_product_paths(text: str) -> list[Path]:
    """Extract displayed final-product paths from a save card."""
    raw = _POSIX_PRODUCT.findall(text) + _WINDOWS_PRODUCT.findall(text)
    unique: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        cleaned = value.strip().rstrip(".,;:)")
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(Path(cleaned))
    return unique


def decide_dialog(
    title: str,
    text: str,
    output_dir: Path,
    expected_names: set[str],
    dialog_kind: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed decision for one visible consent dialog."""
    paths = extract_product_paths(text)
    if dialog_kind != SAVE_DIALOG_KIND and SAVE_TITLE.search(title.strip()) is None:
        return {
            "decision": "deny",
            "reason": "not an expected staged-save dialog",
            "paths": [str(path) for path in paths],
        }
    if len(paths) != 1:
        return {
            "decision": "deny",
            "reason": f"expected exactly one displayed product path, found {len(paths)}",
            "paths": [str(path) for path in paths],
        }
    target = paths[0]
    try:
        parent_ok = target.expanduser().resolve().parent == output_dir.expanduser().resolve()
    except OSError:
        parent_ok = False
    name_ok = target.name in expected_names
    return {
        "decision": "approve" if parent_ok and name_ok else "deny",
        "reason": (
            "expected filename in the exact trial output directory"
            if parent_ok and name_ok
            else f"parent_ok={parent_ok}; expected_name={name_ok}"
        ),
        "paths": [str(target)],
    }


def click_decision(dialog: Any, decision: str) -> None:
    """Click a consent action using semantic classes with text fallbacks."""
    selector = "button.jp-mod-accept" if decision == "approve" else "button.jp-mod-reject"
    button = dialog.locator(selector)
    if button.count():
        button.first.click()
        return
    label = re.compile(
        r"^(保存|允许|save|allow)$" if decision == "approve" else r"^(拒绝|取消|deny|cancel)$",
        re.IGNORECASE,
    )
    dialog.get_by_role("button", name=label).click()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def open_notebook_from_dashboard(page: Any, timeout_ms: int) -> Any:
    """Open JupyterLab from the dashboard, or return an existing Lab page."""
    open_lab = page.locator("#open-lab")
    if not open_lab.count():
        return page

    page.locator("#open-lab[href]:not([href=''])").wait_for(
        state="attached",
        timeout=timeout_ms,
    )
    with page.expect_popup(timeout=timeout_ms) as opened:
        open_lab.click()
    return opened.value


def run_harness(args: argparse.Namespace) -> int:
    from playwright.sync_api import sync_playwright

    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_names = set(args.expected_file or ())
    log_path = Path(args.log).expanduser().resolve()
    ready_file = Path(args.ready_file).expanduser().resolve()
    stop_file = Path(args.stop_file).expanduser().resolve()
    last_dialog_at = time.monotonic()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            channel=args.channel,
            args=[
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-gpu",
                "--renderer-process-limit=1",
            ],
        )
        page = browser.new_page()
        try:
            page.goto(args.notebook_url, wait_until="domcontentloaded", timeout=args.timeout * 1000)
            page = open_notebook_from_dashboard(page, args.timeout * 1000)
            page.wait_for_selector(".jp-Notebook", timeout=args.timeout * 1000)
            ready_file.write_text(
                json.dumps(
                    {
                        "ready_at": datetime.now(UTC).isoformat(),
                        "url": page.url,
                        "expected_files": sorted(expected_names),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            append_jsonl(
                log_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "harness_ready",
                    "url": page.url,
                },
            )

            while True:
                dialogs = page.locator(".jp-Dialog")
                count = dialogs.count()
                if count:
                    last_dialog_at = time.monotonic()
                for index in range(count):
                    dialog = dialogs.nth(index)
                    if not dialog.is_visible():
                        continue
                    header = dialog.locator(".jp-Dialog-header")
                    title = header.inner_text().strip() if header.count() else ""
                    text = dialog.inner_text()
                    marker = dialog.locator(
                        f'[data-peaks-mcp-dialog="{SAVE_DIALOG_KIND}"]'
                    )
                    dialog_kind = SAVE_DIALOG_KIND if marker.count() else None
                    decision = decide_dialog(
                        title,
                        text,
                        output_dir,
                        expected_names,
                        dialog_kind,
                    )
                    append_jsonl(
                        log_path,
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "type": "scoped_save_approval"
                            if decision["decision"] == "approve"
                            else "policy_denial",
                            "title": title,
                            **decision,
                        },
                    )
                    click_decision(dialog, decision["decision"])
                    page.wait_for_timeout(100)

                if stop_file.exists() and time.monotonic() - last_dialog_at >= args.idle_grace:
                    break
                page.wait_for_timeout(200)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                log_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "harness_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        finally:
            browser.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-file", action="append", default=[])
    parser.add_argument("--log", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--idle-grace", type=float, default=1.0)
    return run_harness(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
