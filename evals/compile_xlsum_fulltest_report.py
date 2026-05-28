#!/usr/bin/env python
"""Compile full-test XL-Sum results, bootstrap CIs, and a Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LANGS = ["ind", "swh", "tur"]

MODELS = [
    ("ACROS K32", "acros_k32"),
    ("Frozen BP", "legacy_backpack_360M"),
    ("Gemma 3 270M bf16", "gemma3_270m_bf16"),
    ("Qwen2 0.5B", "qwen2_0b5"),
]

METRICS = [
    ("R-1", "rouge1_f"),
    ("R-2", "rouge2_f"),
    ("R-L", "rougeL_f"),
    ("chrF++", "chrf"),
    ("Pred words", "pred_words"),
    ("Ref words", "ref_words"),
    ("Len ratio", "length_ratio"),
    ("Distinct-1", "distinct_1"),
    ("Distinct-2", "distinct_2"),
    ("Rep-3g", "repeat_3gram_rate"),
    ("Copy-4g", "source_copy_4gram_precision"),
    ("Empty", "empty"),
    ("Label echo", "leading_label_echo"),
]

CORE_METRICS = ["R-L", "chrF++", "Empty", "Rep-3g", "Copy-4g"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("eval_logs/xlsum/full"))
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_logs/significance/xlsum_fulltest_cis_2026-05-14.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("eval_logs/significance/xlsum_fulltest_report_2026-05-14.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    h = hashlib.sha256()
    h.update(str(base).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 0:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize_boot(boot: np.ndarray, point: float, n: int, unit: str) -> dict:
    boot = finite(boot)
    boot.sort()
    return {
        "mean": float(point),
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        "n": int(n),
        "bootstrap_unit": unit,
    }


def bootstrap_values(values: Iterable[float], samples: int, seed: int, chunk_size: int) -> dict:
    vals = finite(values)
    if len(vals) == 0:
        return summarize_boot(np.asarray([], dtype=np.float64), float("nan"), 0, "xlsum_example")
    point = float(vals.mean())
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        size = min(chunk_size, samples - start)
        idx = rng.integers(0, n, size=(size, n), endpoint=False)
        boot[start : start + size] = vals[idx].mean(axis=1)
    return summarize_boot(boot, point, n, "xlsum_example")


def bootstrap_language_balanced(
    values_by_lang: dict[str, Iterable[float]],
    samples: int,
    seed: int,
    chunk_size: int,
) -> dict:
    arrays = {lang: finite(values_by_lang[lang]) for lang in LANGS}
    point = float(sum(arrays[lang].mean() for lang in LANGS) / len(LANGS))
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        size = min(chunk_size, samples - start)
        lang_means = []
        for lang in LANGS:
            vals = arrays[lang]
            n = len(vals)
            idx = rng.integers(0, n, size=(size, n), endpoint=False)
            lang_means.append(vals[idx].mean(axis=1))
        boot[start : start + size] = np.vstack(lang_means).mean(axis=0)
    return {
        **summarize_boot(boot, point, len(LANGS), "language_balanced_xlsum_example"),
        "examples_by_language": {lang: int(len(arrays[lang])) for lang in LANGS},
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def artifact_path(input_dir: Path, prefix: str, lang: str) -> Path:
    filename = f"{prefix}_{lang}_fulltest.jsonl"
    direct = input_dir / filename
    if direct.exists():
        return direct
    return input_dir / "jsonl" / filename


def summary_path(input_dir: Path, prefix: str, lang: str) -> Path:
    filename = f"{prefix}_{lang}_fulltest.summary.json"
    direct = input_dir / filename
    if direct.exists():
        return direct
    return input_dir / "summaries" / filename


def metric_value(summary: dict, metric_label: str) -> dict:
    return summary[metric_label]


def fmt_metric(metric_label: str, value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    if metric_label == "chrF++":
        return f"{value:.2f}"
    if metric_label in {"Pred words", "Ref words"}:
        return f"{value:.1f}"
    return f"{value:.4f}"


def fmt_ci(metric_label: str, stats: dict) -> str:
    return (
        f"{fmt_metric(metric_label, stats['mean'])} "
        f"[{fmt_metric(metric_label, stats['ci95_low'])}, "
        f"{fmt_metric(metric_label, stats['ci95_high'])}]"
    )


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report(payload: dict) -> str:
    lines = [
        "# ACROS XL-Sum Full-Test Rerun Report",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## Protocol",
        "",
        "- Dataset: `csebuetnlp/xlsum`, test split.",
        "- Languages: Indonesian (`ind`), Swahili (`swh`), Turkish (`tur`). Estonian is unavailable in XL-Sum.",
        "- Decoding: completion prompt, greedy decoding, `max_input_tokens=512`, `max_new_tokens=80`.",
        f"- Hardware path: CUDA generation artifacts, compiled locally from `{payload['input_dir']}`.",
        f"- Bootstrap: {payload['bootstrap_samples']} resamples, seed `{payload['seed']}`.",
        "- CIs are example bootstraps within each language and language-balanced bootstraps for aggregate rows.",
        "",
        "## Language-Balanced Aggregate",
        "",
    ]

    aggregate_rows = []
    for model_label, _prefix in MODELS:
        model = payload["models"][model_label]
        row = [model_label]
        for metric_label in CORE_METRICS:
            row.append(fmt_ci(metric_label, metric_value(model["language_balanced"], metric_label)))
        aggregate_rows.append(row)
    lines.append(md_table(["Model", *CORE_METRICS], aggregate_rows))
    lines.extend(["", "## Per-Language Core Results", ""])

    per_lang_rows = []
    for model_label, _prefix in MODELS:
        for lang in LANGS:
            lang_payload = payload["models"][model_label]["by_language"][lang]
            row = [model_label, lang, str(lang_payload["num_examples"])]
            for metric_label in CORE_METRICS:
                row.append(fmt_ci(metric_label, metric_value(lang_payload, metric_label)))
            per_lang_rows.append(row)
    lines.append(md_table(["Model", "Lang", "N", *CORE_METRICS], per_lang_rows))
    lines.extend(["", "## Full Metric Means By Language", ""])

    full_rows = []
    for model_label, _prefix in MODELS:
        for lang in LANGS:
            lang_payload = payload["models"][model_label]["by_language"][lang]
            row = [model_label, lang, str(lang_payload["num_examples"])]
            for metric_label, _metric_key in METRICS:
                row.append(fmt_metric(metric_label, metric_value(lang_payload, metric_label)["mean"]))
            full_rows.append(row)
    lines.append(md_table(["Model", "Lang", "N", *[label for label, _key in METRICS]], full_rows))
    lines.extend(["", "## Artifacts", ""])

    artifact_rows = []
    for model_label, _prefix in MODELS:
        for lang in LANGS:
            lang_payload = payload["models"][model_label]["by_language"][lang]
            artifact_rows.append(
                [
                    model_label,
                    lang,
                    f"`{lang_payload['artifact']}`",
                    f"`{lang_payload['summary_artifact']}`",
                ]
            )
    lines.append(md_table(["Model", "Lang", "JSONL", "Summary"], artifact_rows))
    lines.extend(
        [
            "",
            "Machine-readable CI payload:",
            "",
            f"- `{payload['out_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = {
        "created_by": "evals/compile_xlsum_fulltest_report.py",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "input_dir": str(args.input_dir),
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
        "models": {},
    }

    for model_label, prefix in MODELS:
        by_language = {}
        metric_values_by_lang = {metric_label: {} for metric_label, _key in METRICS}
        for lang in LANGS:
            path = artifact_path(args.input_dir, prefix, lang)
            summary = summary_path(args.input_dir, prefix, lang)
            rows = read_jsonl(path)
            metrics_rows = [row["metrics"] for row in rows]
            by_language[lang] = {
                "artifact": str(path),
                "summary_artifact": str(summary),
                "num_examples": len(metrics_rows),
            }
            for metric_label, metric_key in METRICS:
                vals = [float(row[metric_key]) for row in metrics_rows]
                metric_values_by_lang[metric_label][lang] = vals
                by_language[lang][metric_label] = bootstrap_values(
                    vals,
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, lang, metric_label),
                    args.chunk_size,
                )
        output["models"][model_label] = {
            "prefix": prefix,
            "by_language": by_language,
            "language_balanced": {
                metric_label: bootstrap_language_balanced(
                    values_by_lang,
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, "language_balanced", metric_label),
                    args.chunk_size,
                )
                for metric_label, values_by_lang in metric_values_by_lang.items()
            },
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_report(output), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
