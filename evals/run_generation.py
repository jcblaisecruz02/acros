#!/usr/bin/env python3
"""Unified generation runner for ACROS paper experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASK_TO_SCRIPT = {
    "xlsum": "run_xlsum_generation.py",
    "xlsum_report": "compile_xlsum_fulltest_report.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=TASK_TO_SCRIPT)
    p.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying script")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [sys.executable, str(ROOT / TASK_TO_SCRIPT[args.task]), *args.script_args]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
