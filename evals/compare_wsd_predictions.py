#!/usr/bin/env python
"""Compare two WSD prediction dumps with paired significance tests."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="First WSD result JSON.")
    p.add_argument("--b", type=Path, required=True, help="Second WSD result JSON.")
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument(
        "--a-layer",
        default=None,
        help="For hidden-state result JSONs, compare predictions from this scores_by_layer entry.",
    )
    p.add_argument(
        "--b-layer",
        default=None,
        help="For hidden-state result JSONs, compare predictions from this scores_by_layer entry.",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_predictions(path: Path, layer: str | None) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if layer is not None:
        try:
            return payload["scores_by_layer"][layer]["predictions"]
        except KeyError as exc:
            raise KeyError(f"{path} does not contain predictions for layer {layer!r}") from exc
    if "scores" in payload and "predictions" in payload["scores"]:
        return payload["scores"]["predictions"]
    if "predictions" in payload:
        return payload["predictions"]
    raise KeyError(f"{path} does not contain predictions. Rerun with --include-predictions.")


def prediction_key(item: Mapping) -> str:
    return f"{item['dataset']}::{item['instance_id']}"


def index_predictions(predictions: Iterable[Mapping]) -> Dict[str, Mapping]:
    indexed: Dict[str, Mapping] = {}
    for item in predictions:
        key = prediction_key(item)
        if key in indexed:
            raise ValueError(f"Duplicate prediction key: {key}")
        indexed[key] = item
    return indexed


def log_binom_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = min(a_only, b_only)
    logs = [log_binom_pmf(k, n) for k in range(tail + 1)]
    max_log = max(logs)
    cdf = math.exp(max_log) * sum(math.exp(value - max_log) for value in logs)
    return min(1.0, 2.0 * cdf)


def bootstrap_ci(diffs: Sequence[int], samples: int, seed: int) -> Dict[str, float]:
    if not diffs or samples <= 0:
        return {}
    rng = random.Random(seed)
    n = len(diffs)
    draws: List[float] = []
    for _ in range(samples):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo_idx = int(0.025 * (samples - 1))
    hi_idx = int(0.975 * (samples - 1))
    return {
        "samples": samples,
        "seed": seed,
        "ci95_low": draws[lo_idx],
        "ci95_high": draws[hi_idx],
    }


def main() -> None:
    args = parse_args()
    a_predictions = index_predictions(load_predictions(args.a, args.a_layer))
    b_predictions = index_predictions(load_predictions(args.b, args.b_layer))

    common_keys = sorted(set(a_predictions) & set(b_predictions))
    if not common_keys:
        raise ValueError("No overlapping prediction keys.")

    missing_a = sorted(set(b_predictions) - set(a_predictions))
    missing_b = sorted(set(a_predictions) - set(b_predictions))

    both_correct = a_only = b_only = both_wrong = 0
    diffs: List[int] = []
    disagreements: List[Dict] = []
    for key in common_keys:
        a_item = a_predictions[key]
        b_item = b_predictions[key]
        a_correct = bool(a_item["correct"])
        b_correct = bool(b_item["correct"])
        diffs.append(int(a_correct) - int(b_correct))
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif b_correct and not a_correct:
            b_only += 1
        else:
            both_wrong += 1
        if a_correct != b_correct:
            disagreements.append(
                {
                    "key": key,
                    "target_text": a_item.get("target_text"),
                    "lemma": a_item.get("lemma"),
                    "pos": a_item.get("pos"),
                    "gold_sense_keys": a_item.get("gold_sense_keys"),
                    f"{args.a_name}_prediction": a_item.get("prediction"),
                    f"{args.a_name}_correct": a_correct,
                    f"{args.b_name}_prediction": b_item.get("prediction"),
                    f"{args.b_name}_correct": b_correct,
                }
            )

    n = len(common_keys)
    a_accuracy = (both_correct + a_only) / n
    b_accuracy = (both_correct + b_only) / n
    diff = a_accuracy - b_accuracy
    result = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "a": str(args.a),
        "b": str(args.b),
        "a_layer": args.a_layer,
        "b_layer": args.b_layer,
        "common": n,
        "missing_from_a": len(missing_a),
        "missing_from_b": len(missing_b),
        "a_accuracy": a_accuracy,
        "b_accuracy": b_accuracy,
        "accuracy_delta_a_minus_b": diff,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_pvalue(a_only, b_only),
        "bootstrap_accuracy_delta": bootstrap_ci(diffs, args.bootstrap_samples, args.seed),
        "num_disagreements": len(disagreements),
        "disagreements": disagreements[:100],
    }

    text = (
        f"{args.a_name}: {a_accuracy:.4f} ({both_correct + a_only}/{n})\n"
        f"{args.b_name}: {b_accuracy:.4f} ({both_correct + b_only}/{n})\n"
        f"delta {args.a_name}-{args.b_name}: {diff:.4f}\n"
        f"discordant: {args.a_name}-only={a_only}, {args.b_name}-only={b_only}\n"
        f"McNemar exact p={result['mcnemar_exact_p']:.6g}\n"
    )
    ci = result["bootstrap_accuracy_delta"]
    if ci:
        text += f"paired bootstrap 95% CI: [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]\n"
    print(text, end="")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
