#!/usr/bin/env python3
"""Unified WSD runner for ACROS paper experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASK_TO_SCRIPT = {
    "acros": "score_wsd_gloss_activation.py",
    "gloss_lm": "score_wsd_gloss_lm.py",
    "mfs": "score_wsd_mfs.py",
    "hidden": "score_wsd_hidden_gloss_matching.py",
    "centroid": "score_wsd_centroid.py",
    "compare": "compare_wsd_predictions.py",
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
