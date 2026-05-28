#!/usr/bin/env python
"""Shared helpers for ASTIA V2 SmolLM2 LAMA probes."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


MASK = "[MASK]"
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-360M"


def load_probe(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of examples at {path}")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def answer_token_ids(tokenizer, answer: str) -> List[int]:
    return tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]


def validate_example_shape(ex: Dict[str, Any], index: int | None = None) -> None:
    prefix = f"Example {index}: " if index is not None else ""
    for key in ("prompt", "answer", "answer_token_id", "relation_id"):
        if key not in ex:
            raise ValueError(f"{prefix}missing required key {key!r}")
    if not isinstance(ex["prompt"], str) or not ex["prompt"].strip():
        raise ValueError(f"{prefix}prompt must be a non-empty string")
    if not isinstance(ex["answer"], str) or not ex["answer"].strip():
        raise ValueError(f"{prefix}answer must be a non-empty string")
    if not isinstance(ex["relation_id"], str) or not ex["relation_id"].strip():
        raise ValueError(f"{prefix}relation_id must be a non-empty string")
    if not isinstance(ex["answer_token_id"], int):
        raise ValueError(f"{prefix}answer_token_id must be an integer")


def stratified_sample(
    examples: Sequence[Dict[str, Any]],
    max_examples: int,
    seed: int,
    relation_key: str = "relation_id",
) -> List[Dict[str, Any]]:
    """Deterministically sample up to max_examples while preserving relation mix."""
    if max_examples <= 0 or len(examples) <= max_examples:
        return list(examples)

    import random

    rng = random.Random(seed)
    by_rel: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        by_rel.setdefault(str(ex[relation_key]), []).append(ex)
    for bucket in by_rel.values():
        rng.shuffle(bucket)

    total = len(examples)
    quotas: Dict[str, int] = {}
    fractional = []
    for rel, bucket in by_rel.items():
        exact = len(bucket) * max_examples / total
        base = min(len(bucket), math.floor(exact))
        quotas[rel] = base
        fractional.append((exact - base, rel))

    remaining = max_examples - sum(quotas.values())
    for _, rel in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        if quotas[rel] < len(by_rel[rel]):
            quotas[rel] += 1
            remaining -= 1

    # If many tiny buckets rounded to zero, fill from remaining relation buckets.
    while remaining > 0:
        progressed = False
        for rel in sorted(by_rel):
            if remaining <= 0:
                break
            if quotas[rel] < len(by_rel[rel]):
                quotas[rel] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    sampled = []
    for rel in sorted(by_rel):
        sampled.extend(by_rel[rel][: quotas[rel]])
    rng.shuffle(sampled)
    return sampled[:max_examples]


def relation_counts(examples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(ex["relation_id"]) for ex in examples))


def split_lama_cloze(masked_sentence: str) -> str | None:
    """Return autoregressive prefix before [MASK], or None for unusable rows."""
    if not isinstance(masked_sentence, str) or MASK not in masked_sentence:
        return None
    prefix = masked_sentence.split(MASK, 1)[0].strip()
    # A prefix ending with punctuation usually asks the model to predict a new
    # sentence, not the masked object. Keep natural cloze prefixes only.
    return prefix if prefix else None
