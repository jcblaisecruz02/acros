#!/usr/bin/env python
"""Collect paper-facing paired significance tests into one JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_WSD_COMPARISONS = {
    "acros_k32_vs_base_smollm2_gloss_lm": Path(
        "eval_logs/wsd/compare_acros_k32_vs_base_smollm2_gloss_lm.json"
    ),
    "acros_k32_vs_wordnet_s1": Path("eval_logs/wsd/compare_acros_k32_vs_wordnet_s1.json"),
    "acros_k32_vs_base_smollm2_hidden_l24": Path(
        "eval_logs/wsd/compare_acros_k32_vs_base_smollm2_hidden_l24.json"
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--steering",
        type=Path,
        default=Path("eval_logs/steering/word_level_hidden_coordinate_control.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("eval_logs/significance/paper_significance_tests.json"),
    )
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def log_binom_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)


def exact_two_sided_sign_pvalue(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    logs = [log_binom_pmf(k, n) for k in range(tail + 1)]
    max_log = max(logs)
    cdf = math.exp(max_log) * sum(math.exp(value - max_log) for value in logs)
    return min(1.0, 2.0 * cdf)


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> Dict[str, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2 or samples <= 0:
        return {}
    rng = random.Random(seed)
    n = len(vals)
    means: List[float] = []
    for _ in range(samples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int(0.025 * (samples - 1))
    hi_idx = int(0.975 * (samples - 1))
    return {
        "samples": samples,
        "seed": seed,
        "ci95_low": means[lo_idx],
        "ci95_high": means[hi_idx],
    }


def load_wsd_comparison(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ci = payload.get("bootstrap_accuracy_delta", {})
    return {
        "artifact": str(path),
        "a_name": payload["a_name"],
        "b_name": payload["b_name"],
        "common_instances": payload["common"],
        "a_accuracy": payload["a_accuracy"],
        "b_accuracy": payload["b_accuracy"],
        "accuracy_delta_a_minus_b": payload["accuracy_delta_a_minus_b"],
        "a_only_correct": payload["a_only_correct"],
        "b_only_correct": payload["b_only_correct"],
        "mcnemar_exact_p": payload["mcnemar_exact_p"],
        "bootstrap_accuracy_delta_ci95": {
            "low": ci.get("ci95_low"),
            "high": ci.get("ci95_high"),
            "samples": ci.get("samples"),
            "seed": ci.get("seed"),
        },
    }


def index_rows(rows: Iterable[Mapping]) -> Dict[Tuple[int, float, str], Mapping]:
    out: Dict[Tuple[int, float, str], Mapping] = {}
    for row in rows:
        key = (int(row["case_index"]), float(row["boost"]), str(row["condition"]))
        if key in out:
            raise ValueError(f"Duplicate steering row: {key}")
        out[key] = row
    return out


def paired_metric_test(
    rows: Mapping[Tuple[int, float, str], Mapping],
    condition_a: str,
    condition_b: str,
    metric: str,
    boost: float,
    samples: int,
    seed: int,
) -> Dict:
    keys_a = {case for case, row_boost, cond in rows if row_boost == boost and cond == condition_a}
    keys_b = {case for case, row_boost, cond in rows if row_boost == boost and cond == condition_b}
    common_cases = sorted(keys_a & keys_b)
    diffs: List[float] = []
    a_vals: List[float] = []
    b_vals: List[float] = []
    for case in common_cases:
        a_val = float(rows[(case, boost, condition_a)][metric])
        b_val = float(rows[(case, boost, condition_b)][metric])
        a_vals.append(a_val)
        b_vals.append(b_val)
        diffs.append(a_val - b_val)

    positive = sum(1 for value in diffs if value > 0)
    negative = sum(1 for value in diffs if value < 0)
    ties = len(diffs) - positive - negative
    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)
    mean_diff = sum(diffs) / len(diffs)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "metric": metric,
        "boost": boost,
        "paired_cases": len(diffs),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta_a_minus_b": mean_diff,
        "a_greater": positive,
        "b_greater": negative,
        "ties": ties,
        "two_sided_sign_p": exact_two_sided_sign_pvalue(positive, negative),
        "paired_bootstrap_mean_delta_ci95": bootstrap_mean_ci(diffs, samples, seed),
    }


def steering_tests(path: Path, samples: int, seed: int) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = index_rows(payload["rows"])
    boosts = [float(value) for value in payload["boosts"]]
    comparisons = []
    for boost in boosts:
        comparisons.extend(
            [
                paired_metric_test(
                    rows,
                    "sense_target_best",
                    "sense_random",
                    "delta_target_logprob",
                    boost,
                    samples,
                    seed,
                ),
                paired_metric_test(
                    rows,
                    "sense_target_best",
                    "sense_norm",
                    "delta_target_logprob",
                    boost,
                    samples,
                    seed,
                ),
                paired_metric_test(
                    rows,
                    "sense_target_best",
                    "hidden_target_best",
                    "delta_target_logprob",
                    boost,
                    samples,
                    seed,
                ),
                paired_metric_test(
                    rows,
                    "hidden_target_best",
                    "sense_target_best",
                    "kl_base_to_intervened",
                    boost,
                    samples,
                    seed,
                ),
                paired_metric_test(
                    rows,
                    "hidden_target_best",
                    "hidden_random",
                    "delta_target_logprob",
                    boost,
                    samples,
                    seed,
                ),
                paired_metric_test(
                    rows,
                    "hidden_target_best",
                    "hidden_norm",
                    "delta_target_logprob",
                    boost,
                    samples,
                    seed,
                ),
            ]
        )
    return {
        "artifact": str(path),
        "num_cases": payload["num_cases"],
        "boosts": boosts,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    result = {
        "wsd": {
            name: load_wsd_comparison(path)
            for name, path in DEFAULT_WSD_COMPARISONS.items()
        },
        "steering": steering_tests(args.steering, args.bootstrap_samples, args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
