#!/usr/bin/env python
"""Score a non-oracle CoInCo selector using the model's own top predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from score_coinco_artifact_contribution_selector import (
    candidate_delta_vectors_for_example,
    iter_batches,
    selected_delta_vector_for_example,
)
from score_coinco_lexsub_steering import (
    DEFAULT_MODEL,
    get_lm_weight,
    load_model_and_tokenizer,
    log_metrics,
    parse_boosts,
    summarize_rows,
)


DEFAULT_SOURCE = Path("eval_logs/coinco_lexsub/coinco_test_source_token_k32_targetbest_2026-05-11.json")
DEFAULT_OUT = Path("eval_logs/coinco_lexsub/canary_coinco_self_topk_uniform_2026-05-14.json")
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


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
    p.add_argument("--max-cases", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-pool", type=int, default=256)
    p.add_argument("--variant", choices=["uniform", "weighted"], default="uniform")
    return p.parse_args()


def proxy_candidates(base_logits: torch.Tensor, row: dict, tok, top_k: int, top_pool: int) -> list[dict]:
    original_forms = {str(row.get("wordform", "")).lower(), str(row.get("lemma", "")).lower()}
    top_ids = torch.topk(base_logits.float(), k=min(int(top_pool), base_logits.numel())).indices.tolist()
    candidates = []
    seen = set()
    for token_id in top_ids:
        text = tok.decode([int(token_id)], clean_up_tokenization_spaces=False)
        stripped = text.strip()
        lowered = stripped.lower()
        if not stripped or lowered in original_forms:
            continue
        if token_id in seen:
            continue
        if text and not text[0].isspace():
            continue
        if not WORD_RE.match(stripped):
            continue
        seen.add(int(token_id))
        candidates.append({"text": text, "token_id": int(token_id), "freq": 1, "lemma": stripped})
        if len(candidates) >= int(top_k):
            break
    return candidates


def proxy_log_mass(logits: torch.Tensor, token_ids: list[int], weights: torch.Tensor | None = None) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    ids = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    selected = logp.index_select(-1, ids)
    if weights is None:
        log_weights = torch.full_like(selected, -math.log(float(len(token_ids))))
    else:
        log_weights = weights.to(selected.device).float().clamp_min(1e-12).log()
    return torch.logsumexp(selected + log_weights, dim=-1)


def choose_self_topk_sense(
    base_logits: torch.Tensor,
    deltas: torch.Tensor,
    lm_w: torch.Tensor,
    proxy: list[dict],
    variant: str,
) -> tuple[int, dict]:
    token_ids = [int(item["token_id"]) for item in proxy]
    if variant == "weighted":
        base_logp = F.log_softmax(base_logits.float(), dim=-1)
        ids = torch.tensor(token_ids, device=base_logits.device, dtype=torch.long)
        weights = F.softmax(base_logp.index_select(0, ids), dim=0)
    else:
        weights = None
    base_score = proxy_log_mass(base_logits, token_ids, weights)
    candidate_logits = base_logits.unsqueeze(0) + torch.matmul(deltas.to(lm_w.device), lm_w.T)
    candidate_scores = torch.stack([proxy_log_mass(logits, token_ids, weights) for logits in candidate_logits])
    idx = int(torch.argmax(candidate_scores).item())
    return idx, {
        "proxy_base_log_mass": float(base_score.item()),
        "proxy_selected_log_mass": float(candidate_scores[idx].item()),
        "proxy_delta_log_mass": float((candidate_scores[idx] - base_score).item()),
        "proxy_candidates": proxy,
    }


def score_case_from_batch(
    row: dict,
    out,
    batch_idx: int,
    query_pos: int,
    lm_w: torch.Tensor,
    boosts: list[float],
    tok,
    args: argparse.Namespace,
) -> dict:
    anchor_pos = int(row["anchor_pos"])
    substs = row["usable_substitutes"]
    base_logits = out.logits[batch_idx, query_pos, :].detach().float()
    base_metrics = log_metrics(base_logits, substs)
    first_deltas = candidate_delta_vectors_for_example(
        out, batch_idx, anchor_pos, query_pos, boosts[0], row["intervention_scope"]
    )
    proxy = proxy_candidates(base_logits, row, tok, args.top_k, args.top_pool)
    if not proxy:
        sense_idx = int(torch.argmax(first_deltas.detach().float().norm(dim=-1)).item())
        proxy_info = {"proxy_candidates": [], "proxy_fallback": "contribution_norm_no_candidates"}
    else:
        sense_idx, proxy_info = choose_self_topk_sense(base_logits, first_deltas, lm_w, proxy, args.variant)
        proxy_info["proxy_fallback"] = None

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
            "selector": f"self_topk_{args.variant}",
            "selector_metric": row.get("selector_metric", "weighted_log_mass"),
            "interface": "sense",
            "chosen_sense": int(sense_idx),
            "chosen_hidden_coordinate": None,
            "chosen_hidden_sign": None,
            "base_metrics": base_metrics,
            "boost_results": boost_results,
            "kl_base_to_intervened": first["kl_base_to_intervened"],
            "delta_top_substitute_rank": first["delta_top_substitute_rank"],
            **proxy_info,
        }
    )
    for key in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
        out_row[f"delta_{key}"] = first[f"delta_{key}"]
    return out_row


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

        query_positions = (
            (attention_mask.sum(dim=1) - 1).tolist()
            if attention_mask is not None
            else [input_ids.size(1) - 1] * len(batch_rows)
        )
        for batch_idx, row in enumerate(batch_rows):
            scored.append(score_case_from_batch(row, out, batch_idx, int(query_positions[batch_idx]), lm_w, boosts, tok, args))
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

    selector = f"self_topk_{args.variant}"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "evals/score_coinco_self_proposed_selector.py",
        "method": "coinco_lexsub_self_proposed_selector",
        "source_artifact": str(args.source_artifact),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "split": source.get("split", "test"),
        "selector": selector,
        "selector_metric": source.get("selector_metric", "weighted_log_mass"),
        "variant": args.variant,
        "top_k": args.top_k,
        "top_pool": args.top_pool,
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
