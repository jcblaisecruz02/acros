#!/usr/bin/env python
"""Score factual sharpness on the ASTIA V2 SmolLM2 LAMA probe set."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lama_smollm2_common import DEFAULT_TOKENIZER, load_probe, write_json  # noqa: E402


DEFAULT_MODELS = [
    "smollm2_360m_base=HuggingFaceTB/SmolLM2-360M",
    "backpack_360m_v1=jcblaise/backpack-smollm2-360M",
    "backpack_360m_v3=jcblaise/backpack-smollm2-360M-v3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", default="evals/lama_smollm2.json")
    p.add_argument("--out", default="evals/lama_smollm2_baselines.json")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", action="append", default=None, help="Repeatable label=path spec.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--top-relations", type=int, default=5)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def parse_model_specs(specs: List[str] | None) -> List[tuple[str, str]]:
    specs = specs or DEFAULT_MODELS
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=path, got: {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid model spec: {spec!r}")
        parsed.append((label, path))
    return parsed


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def load_model(model_path: str, device: torch.device, dtype: str, trust_remote_code: bool):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        trust_remote_code=trust_remote_code,
        dtype=resolve_dtype(dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model


def batched(examples: List[dict], batch_size: int):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


@torch.no_grad()
def score_model(
    model,
    tokenizer,
    examples: List[dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
    dtype: str,
    top_relations: int,
) -> Dict:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    n = 0
    top1 = 0
    top5 = 0
    answer_prob_sum = 0.0
    rr_sum = 0.0
    entropy_sum = 0.0
    relation_totals = defaultdict(int)
    relation_top1 = defaultdict(int)

    autocast_enabled = device.type == "cuda" and dtype in {"float16", "bfloat16"}
    autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    for batch in tqdm(list(batched(examples, batch_size)), desc="Scoring LAMA"):
        prompts = [ex["prompt"] for ex in batch]
        answer_ids = torch.tensor([int(ex["answer_token_id"]) for ex in batch], dtype=torch.long, device=device)
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        lengths = enc.attention_mask.sum(dim=1)
        if (lengths <= 0).any():
            raise RuntimeError("Encountered empty tokenized prompt")
        positions = lengths - 1

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits

        next_logits = logits[torch.arange(len(batch), device=device), positions].float()
        log_probs = torch.log_softmax(next_logits, dim=-1)
        probs = log_probs.exp()
        answer_probs = probs[torch.arange(len(batch), device=device), answer_ids]
        top5_ids = torch.topk(next_logits, k=5, dim=-1).indices
        ranks = (next_logits > next_logits[torch.arange(len(batch), device=device), answer_ids].unsqueeze(1)).sum(dim=1) + 1
        entropy = -(probs * log_probs).sum(dim=-1)

        top1_mask = ranks == 1
        top5_mask = (top5_ids == answer_ids.unsqueeze(1)).any(dim=1)

        for i, ex in enumerate(batch):
            rel = str(ex["relation_id"])
            relation_totals[rel] += 1
            relation_top1[rel] += int(top1_mask[i].item())

        n += len(batch)
        top1 += int(top1_mask.sum().item())
        top5 += int(top5_mask.sum().item())
        answer_prob_sum += float(answer_probs.sum().item())
        rr_sum += float((1.0 / ranks.float()).sum().item())
        entropy_sum += float(entropy.sum().item())

    if n == 0:
        raise RuntimeError("No examples to score")

    relation_breakdown = {}
    for rel, total in sorted(relation_totals.items(), key=lambda item: item[1], reverse=True)[:top_relations]:
        relation_breakdown[rel] = {
            "num_examples": total,
            "top1_accuracy": relation_top1[rel] / total,
        }

    return {
        "num_examples": n,
        "top1_accuracy": top1 / n,
        "top5_accuracy": top5 / n,
        "mean_answer_prob": answer_prob_sum / n,
        "mrr": rr_sum / n,
        "output_entropy_mean": entropy_sum / n,
        "top_relations": relation_breakdown,
    }


def main() -> None:
    args = parse_args()
    examples = load_probe(args.probe)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    device = torch.device(args.device)
    results = {}

    for label, model_path in parse_model_specs(args.model):
        print(f"Loading model {label}: {model_path}")
        model = load_model(model_path, device, args.dtype, args.trust_remote_code)
        metrics = score_model(
            model,
            tokenizer,
            examples,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype=args.dtype,
            top_relations=args.top_relations,
        )
        results[label] = {
            "model_path": model_path,
            "metrics": metrics,
        }
        print(
            f"{label}: top1={metrics['top1_accuracy']:.4f} "
            f"top5={metrics['top5_accuracy']:.4f} "
            f"mean_prob={metrics['mean_answer_prob']:.6f} "
            f"mrr={metrics['mrr']:.4f} entropy={metrics['output_entropy_mean']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "probe": args.probe,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "num_probe_examples_scored": len(examples),
        "results": results,
    }
    write_json(args.out, payload)
    print(f"Wrote scores to {args.out}")


if __name__ == "__main__":
    main()
