#!/usr/bin/env python
"""Compile example-level bootstrap CIs for rerun SENSiA intrinsic artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence


LANGS = ["est", "ind", "swh", "tur"]
MODELS = {
    "Unadapted": "acros_k32_base_eng",
    "Frozen BP": "legacy_backpack_360M",
    "ACROS": "acros_k32",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--details-dir", type=Path, default=Path("eval_logs/post_adapt/details"))
    p.add_argument("--out", type=Path, default=Path("eval_logs/significance/sensia_intrinsic_rerun_cis_2026-05-14.json"))
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260514)
    return p.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    h = hashlib.sha256()
    h.update(str(base).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize_boot(values: list[float], point: float) -> dict:
    values.sort()
    return {
        "mean": point,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def load_payload(details_dir: Path, tag: str, lang: str) -> dict:
    path = details_dir / f"rerun_{tag}_{lang}_2026-05-14.details.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"] = str(path)
    return payload


def point_from_records(records: list[dict], metric: str) -> float:
    if metric == "ctx@1":
        vals = [
            0.5 * (float(row["ctx_src2tgt_correct"]) + float(row["ctx_tgt2src_correct"]))
            for row in records
        ]
        return sum(vals) / len(vals)
    if metric == "sns@1":
        vals = [
            0.5 * (float(row["sns_src2tgt_correct"]) + float(row["sns_tgt2src_correct"]))
            for row in records
        ]
        return sum(vals) / len(vals)
    if metric == "PPL":
        nll = sum(float(row["target_nll"]) for row in records)
        toks = sum(int(row["target_tokens"]) for row in records)
        return math.exp(nll / toks)
    raise KeyError(metric)


def metric_for_indices(records: list[dict], indices: list[int], metric: str) -> float:
    if metric == "ctx@1":
        return sum(
            0.5
            * (
                float(records[idx]["ctx_src2tgt_correct"])
                + float(records[idx]["ctx_tgt2src_correct"])
            )
            for idx in indices
        ) / len(indices)
    if metric == "sns@1":
        return sum(
            0.5
            * (
                float(records[idx]["sns_src2tgt_correct"])
                + float(records[idx]["sns_tgt2src_correct"])
            )
            for idx in indices
        ) / len(indices)
    if metric == "PPL":
        nll = sum(float(records[idx]["target_nll"]) for idx in indices)
        toks = sum(int(records[idx]["target_tokens"]) for idx in indices)
        return math.exp(nll / toks)
    raise KeyError(metric)


def bootstrap_language(records: list[dict], metric: str, samples: int, seed: int) -> dict:
    point = point_from_records(records, metric)
    rng = random.Random(seed)
    n = len(records)
    boot = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        boot.append(metric_for_indices(records, indices, metric))
    result = summarize_boot(boot, point)
    result["n"] = n
    result["bootstrap_unit"] = "flores_sentence_pair"
    return result


def bootstrap_language_balanced(
    records_by_lang: dict[str, list[dict]],
    metric: str,
    samples: int,
    seed: int,
) -> dict:
    point = sum(point_from_records(records_by_lang[lang], metric) for lang in LANGS) / len(LANGS)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        lang_values = []
        for lang in LANGS:
            records = records_by_lang[lang]
            n = len(records)
            indices = [rng.randrange(n) for _ in range(n)]
            lang_values.append(metric_for_indices(records, indices, metric))
        boot.append(sum(lang_values) / len(lang_values))
    result = summarize_boot(boot, point)
    result["num_languages"] = len(LANGS)
    result["bootstrap_unit"] = "language_balanced_flores_sentence_pair"
    return result


def main() -> None:
    args = parse_args()
    output = {
        "created_by": "evals/compile_sensia_intrinsic_rerun_cis.py",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "models": {},
    }
    for model_label, tag in MODELS.items():
        records_by_lang = {}
        by_language = {}
        for lang in LANGS:
            payload = load_payload(args.details_dir, tag, lang)
            records = payload["records"]
            records_by_lang[lang] = records
            by_language[lang] = {
                "artifact": payload["artifact"],
                "summary": payload["summary"],
                "num_examples": len(records),
                "ctx@1": bootstrap_language(
                    records,
                    "ctx@1",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, lang, "ctx"),
                ),
                "sns@1": bootstrap_language(
                    records,
                    "sns@1",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, lang, "sns"),
                ),
                "PPL": bootstrap_language(
                    records,
                    "PPL",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, lang, "ppl"),
                ),
            }
        output["models"][model_label] = {
            "tag": tag,
            "by_language": by_language,
            "language_balanced": {
                "ctx@1": bootstrap_language_balanced(
                    records_by_lang,
                    "ctx@1",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, "balanced", "ctx"),
                ),
                "sns@1": bootstrap_language_balanced(
                    records_by_lang,
                    "sns@1",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, "balanced", "sns"),
                ),
                "PPL": bootstrap_language_balanced(
                    records_by_lang,
                    "PPL",
                    args.bootstrap_samples,
                    stable_seed(args.seed, model_label, "balanced", "ppl"),
                ),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
