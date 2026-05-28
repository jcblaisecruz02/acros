#!/usr/bin/env python
"""Score a non-oracle contribution-norm selector on stored CoInCo prompts."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from score_coinco_lexsub_steering import (
    DEFAULT_MODEL,
    candidate_delta_vectors,
    get_lm_weight,
    load_model_and_tokenizer,
    log_metrics,
    parse_boosts,
    selected_delta_vector,
    summarize_rows,
)


DEFAULT_SOURCE = Path("eval_logs/coinco_lexsub/coinco_test_source_token_k32_targetbest_2026-05-11.json")
DEFAULT_OUT = Path("eval_logs/coinco_lexsub/coinco_test_source_token_k32_contribution_norm_2026-05-14.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--boosts", default="1.2,1.5,2.0,3.0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def gate_for_example(out, batch_idx: int, delta: torch.Tensor) -> torch.Tensor:
    if not hasattr(out, "effective_gate") or out.effective_gate is None:
        return torch.ones((), device=delta.device, dtype=delta.dtype)
    gate = out.effective_gate.detach().float().to(delta.device)
    if gate.numel() == 1:
        return gate.reshape(())
    if gate.shape[0] == delta.shape[0]:
        return gate
    if gate.shape[0] > batch_idx:
        return gate[batch_idx]
    return gate.reshape(-1)[0]


def candidate_delta_vectors_for_example(out, batch_idx: int, anchor_pos: int, query_pos: int, boost: float, scope: str):
    contextualization = out.contextualization.detach().float()
    senses = out.senses.detach().float()
    scale = float(boost) - 1.0

    if scope == "sense_axis":
        per_sense = torch.matmul(contextualization[batch_idx : batch_idx + 1], senses[batch_idx : batch_idx + 1])
        delta = scale * per_sense[0, :, query_pos, :]
    else:
        weights = contextualization[batch_idx, :, query_pos, anchor_pos].unsqueeze(-1)
        source_senses = senses[batch_idx, :, anchor_pos, :]
        delta = scale * weights * source_senses

    return gate_for_example(out, batch_idx, delta) * delta


def selected_delta_vector_for_example(
    out, batch_idx: int, anchor_pos: int, query_pos: int, sense_idx: int, boost: float, scope: str
):
    return candidate_delta_vectors_for_example(out, batch_idx, anchor_pos, query_pos, boost, scope)[sense_idx]


def score_stored_case_from_batch(
    row: dict,
    out,
    batch_idx: int,
    query_pos: int,
    lm_w: torch.Tensor,
    boosts: list[float],
) -> dict:
    anchor_pos = int(row["anchor_pos"])
    substs = row["usable_substitutes"]

    base_logits = out.logits[batch_idx, query_pos, :].detach().float()
    base_metrics = log_metrics(base_logits, substs)
    first_deltas = candidate_delta_vectors_for_example(
        out, batch_idx, anchor_pos, query_pos, boosts[0], row["intervention_scope"]
    )
    sense_idx = int(torch.argmax(first_deltas.detach().float().norm(dim=-1)).item())

    base_p = F.softmax(base_logits, dim=-1)
    base_logp = F.log_softmax(base_logits, dim=-1)
    boost_results = []
    for boost in boosts:
        delta = selected_delta_vector_for_example(
            out, batch_idx, anchor_pos, query_pos, sense_idx, boost, row["intervention_scope"]
        )
        mod_logits = base_logits + torch.matmul(delta.to(lm_w.device), lm_w.T)
        mod_metrics = log_metrics(mod_logits, substs)
        mod_logp = F.log_softmax(mod_logits.float(), dim=-1)
        result = {
            "sense_boost": float(boost),
            "kl_base_to_intervened": float((base_p * (base_logp - mod_logp)).sum().item()),
        }
        for key in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
            result[f"{key}_intervened"] = mod_metrics[key]
            result[f"delta_{key}"] = mod_metrics[key] - base_metrics[key]
        result["top_substitute_rank_intervened"] = mod_metrics["top_substitute_rank"]
        result["delta_top_substitute_rank"] = base_metrics["top_substitute_rank"] - mod_metrics["top_substitute_rank"]
        boost_results.append(result)

    first = boost_results[0]
    out_row = {
        key: row.get(key)
        for key in [
            "id",
            "split",
            "MASCfile",
            "MASCsentID",
            "source_token_id",
            "wordform",
            "lemma",
            "posTT",
            "posMASC",
            "prompt",
            "anchor_pos",
            "intervention_scope",
            "sense_budget_scope",
            "usable_substitutes",
        ]
    }
    out_row.update(
        {
            "selector": "contribution_norm",
            "selector_metric": row.get("selector_metric", "weighted_log_mass"),
            "interface": "sense",
            "chosen_sense": sense_idx,
            "chosen_hidden_coordinate": None,
            "chosen_hidden_sign": None,
            "base_metrics": base_metrics,
            "boost_results": boost_results,
            "kl_base_to_intervened": first["kl_base_to_intervened"],
            "delta_top_substitute_rank": first["delta_top_substitute_rank"],
        }
    )
    for key in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
        out_row[f"delta_{key}"] = first[f"delta_{key}"]
    return out_row


def iter_batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), max(1, int(batch_size))):
        yield start, rows[start : start + max(1, int(batch_size))]


def score_batches(rows: list[dict], model, tok, lm_w: torch.Tensor, args: argparse.Namespace, boosts: list[float]):
    scored = []
    for start, batch_rows in iter_batches(rows, args.batch_size):
        prompts = [row["prompt"] for row in batch_rows]
        enc = tok(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

        query_positions = (attention_mask.sum(dim=1) - 1).tolist() if attention_mask is not None else [input_ids.size(1) - 1] * len(batch_rows)
        for batch_idx, row in enumerate(batch_rows):
            scored.append(score_stored_case_from_batch(row, out, batch_idx, int(query_positions[batch_idx]), lm_w, boosts))
        if len(scored) % 100 <= len(batch_rows):
            print(f"scored {len(scored)}/{len(rows)}", flush=True)
    return scored


def main() -> None:
    args = parse_args()
    boosts = parse_boosts(args.boosts)
    source = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    rows = list(source["cases"])
    if args.max_cases and len(rows) > args.max_cases:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_cases)
        rows.sort(key=lambda item: int(item.get("source_token_id") or 0))

    model_args = SimpleNamespace(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    model, tok = load_model_and_tokenizer(model_args)
    lm_w = get_lm_weight(model).to(args.device)

    scored = score_batches(rows, model, tok, lm_w, args, boosts)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "evals/score_coinco_artifact_contribution_selector.py",
        "method": "coinco_lexsub_sense_steering_from_stored_prompts",
        "source_artifact": str(args.source_artifact),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "split": source.get("split", "test"),
        "selector": "contribution_norm",
        "selector_metric": source.get("selector_metric", "weighted_log_mass"),
        "interface": "sense",
        "intervention_scope": source.get("intervention_scope", "source_token"),
        "boosts": boosts,
        "seed": args.seed,
        "max_cases": args.max_cases,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": args.batch_size,
        "summary": summarize_rows(scored, boosts, source.get("selector_metric", "weighted_log_mass")),
        "cases": scored,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
