#!/usr/bin/env python
"""Build the ASTIA V2 SmolLM2-filtered LAMA T-REx factual probe set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lama_smollm2_common import (  # noqa: E402
    DEFAULT_TOKENIZER,
    answer_token_ids,
    relation_counts,
    split_lama_cloze,
    stratified_sample,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="facebook/lama")
    p.add_argument("--config", default="trex")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out", default="evals/lama_smollm2.json")
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--min-examples", type=int, default=200)
    p.add_argument("--raw-limit", type=int, default=0, help="Optional source-row cap for smoke tests.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def iter_rows(ds, raw_limit: int) -> Iterator[Dict]:
    if raw_limit and raw_limit > 0:
        for i in range(min(raw_limit, len(ds))):
            yield ds[i]
    else:
        yield from ds


def build_examples(args: argparse.Namespace) -> tuple[list[dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
    )

    stats = {
        "raw_rows_seen": 0,
        "missing_mask_or_prompt": 0,
        "missing_answer": 0,
        "multi_token_answer": 0,
        "duplicate_rows": 0,
        "kept_before_sampling": 0,
    }
    examples = []
    seen = set()

    total = min(args.raw_limit, len(ds)) if args.raw_limit and args.raw_limit > 0 else len(ds)
    for row in tqdm(iter_rows(ds, args.raw_limit), total=total, desc="Filtering LAMA T-REx"):
        stats["raw_rows_seen"] += 1

        prompt = split_lama_cloze(row.get("masked_sentence", ""))
        if prompt is None:
            stats["missing_mask_or_prompt"] += 1
            continue

        answer = str(row.get("obj_label", "")).strip()
        if not answer:
            stats["missing_answer"] += 1
            continue

        ids = answer_token_ids(tokenizer, answer)
        if len(ids) != 1:
            stats["multi_token_answer"] += 1
            continue

        relation_id = str(row.get("predicate_id", "")).strip()
        if not relation_id:
            relation_id = "unknown"

        key = (prompt, answer, relation_id)
        if args.dedupe and key in seen:
            stats["duplicate_rows"] += 1
            continue
        seen.add(key)

        examples.append(
            {
                "prompt": prompt,
                "answer": answer,
                "answer_token_id": int(ids[0]),
                "relation_id": relation_id,
            }
        )

    stats["kept_before_sampling"] = len(examples)
    sampled = stratified_sample(examples, args.max_examples, args.seed)
    stats["kept_after_sampling"] = len(sampled)
    stats["num_relations_before_sampling"] = len(relation_counts(examples))
    stats["num_relations_after_sampling"] = len(relation_counts(sampled))
    stats["relation_counts_after_sampling"] = relation_counts(sampled)

    if len(sampled) < args.min_examples:
        raise RuntimeError(
            f"Only kept {len(sampled)} examples, below --min-examples={args.min_examples}. "
            "Increase --raw-limit, relax filtering, or inspect the dataset/tokenizer."
        )

    return sampled, stats


def main() -> None:
    args = parse_args()
    examples, stats = build_examples(args)
    out_path = Path(args.out)
    write_json(out_path, examples)

    meta_path = out_path.with_suffix(".meta.json")
    write_json(
        meta_path,
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "tokenizer": args.tokenizer,
            "max_examples": args.max_examples,
            "raw_limit": args.raw_limit,
            "seed": args.seed,
            "stats": stats,
        },
    )

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote probe set to {out_path}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
