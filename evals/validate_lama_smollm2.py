#!/usr/bin/env python
"""Validate the ASTIA V2 SmolLM2 LAMA probe JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lama_smollm2_common import (  # noqa: E402
    DEFAULT_TOKENIZER,
    answer_token_ids,
    load_probe,
    relation_counts,
    validate_example_shape,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", default="evals/lama_smollm2.json")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_probe(args.probe)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    seen = set()
    for i, ex in enumerate(examples):
        validate_example_shape(ex, i)
        ids = answer_token_ids(tokenizer, ex["answer"])
        if len(ids) != 1:
            raise ValueError(f"Example {i}: answer {ex['answer']!r} tokenizes to {ids}, expected one token")
        if int(ids[0]) != int(ex["answer_token_id"]):
            raise ValueError(
                f"Example {i}: stored answer_token_id={ex['answer_token_id']} "
                f"but tokenizer returned {ids[0]}"
            )
        key = (ex["prompt"], ex["answer"], ex["relation_id"])
        if key in seen:
            raise ValueError(f"Example {i}: duplicate prompt/answer/relation row: {key!r}")
        seen.add(key)

    counts = relation_counts(examples)
    print(f"Validated {len(examples)} examples from {args.probe}")
    print(f"Relations: {len(counts)}")
    print("Top relations:")
    for rel, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {rel}: {count}")


if __name__ == "__main__":
    main()
