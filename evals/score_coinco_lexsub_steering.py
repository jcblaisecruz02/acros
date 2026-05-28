#!/usr/bin/env python
"""Score ACROS sense steering on CoInCo lexical-substitution cases."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_MODEL = "jcblaise/sense-smollm2-360M-k32"
DEFAULT_CASES = Path("evals/data/coinco/coinco_lexsub_cases.json")
DEFAULT_OUT = Path("eval_logs/coinco_lexsub/coinco_lexsub_steering.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--boosts", default="1.2,1.5,2.0,3.0")
    p.add_argument(
        "--selector",
        choices=[
            "target_best",
            "norm",
            "contribution_norm",
            "random",
            "hidden_target_best",
            "hidden_norm",
            "hidden_random",
        ],
        default="target_best",
    )
    p.add_argument(
        "--selector-metric",
        choices=["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"],
        default="weighted_log_mass",
    )
    p.add_argument(
        "--intervention-scope",
        choices=["source_token", "sense_axis"],
        default="source_token",
        help=(
            "source_token boosts the chosen sense contribution from the CoInCo target token "
            "to the answer position; sense_axis matches the older equivalence probe and boosts "
            "the selected sense axis contribution into the answer position."
        ),
    )
    p.add_argument("--min-substitute-freq", type=int, default=1)
    p.add_argument("--exclude-problematic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--one-token-substitutes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--coord-chunk-size", type=int, default=128)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def parse_boosts(spec: str) -> List[float]:
    boosts = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if not boosts:
        raise ValueError("--boosts produced no values")
    return boosts


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def bootstrap_mean_ci(xs: Sequence[float], n: int = 1000, seed: int = 13) -> tuple[float, float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return (boots[int(0.025 * n)], boots[int(0.975 * n)])


def find_subseq(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def build_prompt(case: dict) -> str:
    sentence = " ".join(str(case["sentence"]).split())
    word = str(case["wordform"])
    return f'Context: {sentence}\nA context-appropriate one-word substitute for "{word}" is'


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = args.tokenizer or args.model
    local_files_only = bool(getattr(args, "local_files_only", False))
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=cfg,
        trust_remote_code=args.trust_remote_code,
        dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.to(torch.device(args.device))
    model.eval()
    return model, tok


def get_lm_weight(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().float()
    out_emb = model.get_output_embeddings()
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise ValueError("Model has no output embedding / LM head weight")
    return out_emb.weight.detach().float()


def usable_substitutes(case: dict, tok, args: argparse.Namespace) -> List[dict]:
    target_forms = {str(case.get("wordform", "")).lower(), str(case.get("lemma", "")).lower()}
    out = []
    seen = set()
    for subst in case.get("substitutes", []):
        lemma = " ".join(str(subst.get("lemma", "")).split())
        if not lemma or lemma.lower() in target_forms:
            continue
        if int(subst.get("freq", 1)) < args.min_substitute_freq:
            continue
        text = " " + lemma
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if args.one_token_substitutes and len(ids) != 1:
            continue
        if not ids:
            continue
        token_id = int(ids[0])
        key = (token_id, lemma.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "lemma": lemma,
                "text": text,
                "token_id": token_id,
                "freq": int(subst.get("freq", 1)),
                "coinco_pos": subst.get("pos", ""),
            }
        )
    return out


def log_metrics(logits: torch.Tensor, substs: List[dict]) -> Dict[str, float]:
    token_ids = torch.tensor([s["token_id"] for s in substs], device=logits.device, dtype=torch.long)
    freqs = torch.tensor([max(1, int(s["freq"])) for s in substs], device=logits.device, dtype=torch.float32)
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.index_select(-1, token_ids)
    log_weights = (freqs / freqs.sum()).log()
    top_idx = int(torch.argmax(freqs).item())
    top_id = int(token_ids[top_idx].item())
    top_logprob = float(logp[top_id].item())
    top_rank = int((logits > logits[top_id]).sum().item() + 1)
    return {
        "weighted_log_mass": float(torch.logsumexp(selected + log_weights, dim=-1).item()),
        "unweighted_log_mass": float(torch.logsumexp(selected, dim=-1).item()),
        "top_substitute_logprob": top_logprob,
        "top_substitute_rank": top_rank,
        "num_usable_substitutes": len(substs),
        "total_substitute_freq": int(freqs.sum().item()),
    }


def batched_log_metrics(logits: torch.Tensor, substs: List[dict]) -> Dict[str, torch.Tensor]:
    token_ids = torch.tensor([s["token_id"] for s in substs], device=logits.device, dtype=torch.long)
    freqs = torch.tensor([max(1, int(s["freq"])) for s in substs], device=logits.device, dtype=torch.float32)
    log_z = torch.logsumexp(logits.float(), dim=-1)
    selected = logits.float().index_select(-1, token_ids)
    log_weights = (freqs / freqs.sum()).log()
    top_idx = int(torch.argmax(freqs).item())
    top_id = int(token_ids[top_idx].item())
    return {
        "weighted_log_mass": torch.logsumexp(selected + log_weights, dim=-1) - log_z,
        "unweighted_log_mass": torch.logsumexp(selected, dim=-1) - log_z,
        "top_substitute_logprob": selected[:, top_idx] - log_z,
        "top_substitute_rank": (logits.float() > logits.float()[:, top_id : top_id + 1]).sum(dim=-1) + 1,
    }


def candidate_delta_vectors(out, anchor_pos: int, boost: float, scope: str) -> torch.Tensor:
    contextualization = out.contextualization.detach().float()
    senses = out.senses.detach().float()
    last_pos = contextualization.size(2) - 1
    scale = float(boost) - 1.0

    if scope == "sense_axis":
        per_sense = torch.matmul(contextualization, senses)
        delta = scale * per_sense[0, :, last_pos, :]
    else:
        weights = contextualization[0, :, last_pos, anchor_pos].unsqueeze(-1)
        source_senses = senses[0, :, anchor_pos, :]
        delta = scale * weights * source_senses

    if hasattr(out, "effective_gate") and out.effective_gate is not None:
        gate = out.effective_gate.detach().float().to(delta.device)
        delta = gate * delta
    return delta


def selected_delta_vector(out, anchor_pos: int, sense_idx: int, boost: float, scope: str) -> torch.Tensor:
    return candidate_delta_vectors(out, anchor_pos, boost, scope)[sense_idx]


def hidden_last_from_output(out) -> torch.Tensor:
    if hasattr(out, "hidden_states") and isinstance(out.hidden_states, torch.Tensor):
        return out.hidden_states[0, -1, :].detach().float()
    if (
        hasattr(out, "base_hidden_states")
        and out.base_hidden_states is not None
        and hasattr(out, "sense_mix")
        and out.sense_mix is not None
        and hasattr(out, "effective_gate")
        and out.effective_gate is not None
    ):
        gate = out.effective_gate.detach().float().to(out.sense_mix.device)
        hidden = out.base_hidden_states.detach().float() + gate * out.sense_mix.detach().float()
        return hidden[0, -1, :].detach().float()
    raise ValueError("Hidden-coordinate controls require model outputs with combined hidden states.")


def choose_hidden_target_coordinate(
    base_logits: torch.Tensor,
    lm_w: torch.Tensor,
    substs: List[dict],
    selector_metric: str,
    budget: float,
    chunk_size: int,
) -> tuple[int, float]:
    if budget <= 0:
        return 0, 1.0
    best_score = float("-inf")
    best_coord = 0
    best_sign = 1.0
    hidden_dim = int(lm_w.size(1))
    for start in range(0, hidden_dim, int(chunk_size)):
        stop = min(hidden_dim, start + int(chunk_size))
        cols = lm_w[:, start:stop].T
        plus_logits = base_logits.unsqueeze(0) + float(budget) * cols
        plus_scores = batched_log_metrics(plus_logits, substs)[selector_metric]
        plus_val, plus_idx = torch.max(plus_scores, dim=0)
        if float(plus_val.item()) > best_score:
            best_score = float(plus_val.item())
            best_coord = start + int(plus_idx.item())
            best_sign = 1.0

        minus_logits = base_logits.unsqueeze(0) - float(budget) * cols
        minus_scores = batched_log_metrics(minus_logits, substs)[selector_metric]
        minus_val, minus_idx = torch.max(minus_scores, dim=0)
        if float(minus_val.item()) > best_score:
            best_score = float(minus_val.item())
            best_coord = start + int(minus_idx.item())
            best_sign = -1.0
    return best_coord, best_sign


def target_best_sense_idx_for_budget(
    out,
    base_logits: torch.Tensor,
    lm_w: torch.Tensor,
    substs: List[dict],
    anchor_pos: int,
    boost: float,
    scope: str,
    selector_metric: str,
) -> int:
    deltas = candidate_delta_vectors(out, anchor_pos, boost, scope)
    candidate_logits = base_logits.unsqueeze(0) + torch.matmul(deltas.to(lm_w.device), lm_w.T)
    cand_metrics = batched_log_metrics(candidate_logits, substs)
    return int(torch.argmax(cand_metrics[selector_metric]).item())


def score_case(case: dict, model, tok, lm_w: torch.Tensor, args: argparse.Namespace, boosts: List[float], rng: random.Random):
    prompt = build_prompt(case)
    substs = usable_substitutes(case, tok, args)
    if not substs:
        return None, "no_usable_substitutes"

    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    anchor_ids = tok(str(case["wordform"]), add_special_tokens=False)["input_ids"]
    seq = input_ids[0].tolist()
    anchor_start = find_subseq(seq, anchor_ids)
    if anchor_start < 0:
        return None, "anchor_not_found"
    anchor_pos = anchor_start + len(anchor_ids) - 1

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    if not hasattr(out, "senses") or out.senses is None or not hasattr(out, "contextualization"):
        raise ValueError("CoInCo steering requires model outputs with senses and contextualization")

    base_logits = out.logits[0, -1, :].detach().float()
    base_metrics = log_metrics(base_logits, substs)
    is_hidden_selector = args.selector.startswith("hidden_")

    if args.selector == "random":
        sense_idx = rng.randrange(int(out.senses.shape[1]))
    elif args.selector == "norm":
        norms = out.senses[0, :, anchor_pos, :].detach().float().norm(dim=-1)
        sense_idx = int(torch.argmax(norms).item())
    elif args.selector == "contribution_norm":
        deltas = candidate_delta_vectors(out, anchor_pos, boosts[0], args.intervention_scope)
        sense_idx = int(torch.argmax(deltas.detach().float().norm(dim=-1)).item())
    elif args.selector == "target_best":
        sense_idx = target_best_sense_idx_for_budget(
            out, base_logits, lm_w, substs, anchor_pos, boosts[0], args.intervention_scope, args.selector_metric
        )
    else:
        sense_idx = target_best_sense_idx_for_budget(
            out, base_logits, lm_w, substs, anchor_pos, boosts[0], args.intervention_scope, args.selector_metric
        )
        hidden_last = hidden_last_from_output(out).to(lm_w.device)
        first_budget = float(selected_delta_vector(out, anchor_pos, sense_idx, boosts[0], args.intervention_scope).norm().item())
        if args.selector == "hidden_random":
            hidden_coord = rng.randrange(int(hidden_last.numel()))
            hidden_sign = 1.0 if rng.random() < 0.5 else -1.0
        elif args.selector == "hidden_norm":
            hidden_coord = int(torch.argmax(hidden_last.abs()).item())
            hidden_sign = 1.0 if float(hidden_last[hidden_coord].item()) >= 0 else -1.0
        else:
            hidden_coord, hidden_sign = choose_hidden_target_coordinate(
                base_logits, lm_w, substs, args.selector_metric, first_budget, args.coord_chunk_size
            )

    base_p = F.softmax(base_logits, dim=-1)
    base_logp = F.log_softmax(base_logits, dim=-1)
    boost_results = []
    for boost in boosts:
        if is_hidden_selector:
            budget = float(selected_delta_vector(out, anchor_pos, sense_idx, boost, args.intervention_scope).norm().item())
            mod_logits = base_logits + float(hidden_sign) * budget * lm_w[:, int(hidden_coord)]
        else:
            delta = selected_delta_vector(out, anchor_pos, sense_idx, boost, args.intervention_scope)
            mod_logits = base_logits + torch.matmul(delta.to(lm_w.device), lm_w.T)
        mod_metrics = log_metrics(mod_logits, substs)
        mod_logp = F.log_softmax(mod_logits.float(), dim=-1)
        kl = float((base_p * (base_logp - mod_logp)).sum().item())
        row = {
            "sense_boost": float(boost),
            "kl_base_to_intervened": kl,
        }
        if is_hidden_selector:
            row["matched_budget_source"] = "source_token_target_best_sense_delta_norm"
            row["hidden_delta_norm"] = budget
        for key in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
            row[f"{key}_intervened"] = mod_metrics[key]
            row[f"delta_{key}"] = mod_metrics[key] - base_metrics[key]
        row["top_substitute_rank_intervened"] = mod_metrics["top_substitute_rank"]
        row["delta_top_substitute_rank"] = base_metrics["top_substitute_rank"] - mod_metrics["top_substitute_rank"]
        boost_results.append(row)

    first = boost_results[0]
    out_row = {
        "id": case["id"],
        "split": case["split"],
        "MASCfile": case.get("MASCfile", ""),
        "MASCsentID": case.get("MASCsentID", ""),
        "source_token_id": case.get("source_token_id", ""),
        "wordform": case["wordform"],
        "lemma": case.get("lemma", ""),
        "posTT": case.get("posTT", ""),
        "posMASC": case.get("posMASC", ""),
        "prompt": prompt,
        "anchor_pos": int(anchor_pos),
        "selector": args.selector,
        "selector_metric": args.selector_metric,
        "interface": "hidden_coordinate" if is_hidden_selector else "sense",
        "intervention_scope": "hidden_coordinate" if is_hidden_selector else args.intervention_scope,
        "sense_budget_scope": args.intervention_scope if is_hidden_selector else None,
        "chosen_sense": int(sense_idx),
        "chosen_hidden_coordinate": int(hidden_coord) if is_hidden_selector else None,
        "chosen_hidden_sign": float(hidden_sign) if is_hidden_selector else None,
        "base_metrics": base_metrics,
        "usable_substitutes": substs,
        "boost_results": boost_results,
    }
    for key in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
        out_row[f"delta_{key}"] = first[f"delta_{key}"]
    out_row["kl_base_to_intervened"] = first["kl_base_to_intervened"]
    out_row["delta_top_substitute_rank"] = first["delta_top_substitute_rank"]
    return out_row, None


def choose_cases(payload: dict, args: argparse.Namespace, tok) -> tuple[List[dict], dict]:
    cases = payload["cases"]
    if args.split != "all":
        cases = [c for c in cases if c.get("split") == args.split]
    if args.exclude_problematic:
        cases = [c for c in cases if c.get("problematic") != "yes"]

    eligible = []
    filter_counts: Counter[str] = Counter()
    for case in cases:
        substs = usable_substitutes(case, tok, args)
        if not substs:
            filter_counts["no_usable_substitutes"] += 1
            continue
        eligible.append(case)

    rng = random.Random(args.seed)
    pre_sample_eligible = len(eligible)
    if args.max_cases and args.max_cases > 0 and len(eligible) > args.max_cases:
        eligible = rng.sample(eligible, args.max_cases)
        eligible.sort(key=lambda c: int(c.get("source_token_id", "0")))

    return eligible, {
        "input_cases_after_split_problematic_filter": len(cases),
        "eligible_cases_after_tokenizer_filter": pre_sample_eligible,
        "selected_cases_after_sampling": len(eligible),
        "pre_sample_no_usable_substitutes": int(filter_counts["no_usable_substitutes"]),
    }


def summarize_rows(rows: List[dict], boosts: List[float], selector_metric: str) -> dict:
    summary = {
        "num_cases": len(rows),
        "selector_metric": selector_metric,
        "first_boost": boosts[0],
    }
    for metric in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
        deltas = [r[f"delta_{metric}"] for r in rows]
        summary[f"mean_delta_{metric}"] = mean(deltas)
        summary[f"mean_delta_{metric}_ci95"] = list(bootstrap_mean_ci(deltas))
        summary[f"success_rate_delta_{metric}_gt_0"] = mean([1.0 if d > 0 else 0.0 for d in deltas])
    kls = [r["kl_base_to_intervened"] for r in rows]
    rank_deltas = [r["delta_top_substitute_rank"] for r in rows]
    summary["mean_kl_base_to_intervened"] = mean(kls)
    summary["mean_kl_base_to_intervened_ci95"] = list(bootstrap_mean_ci(kls))
    summary["mean_delta_top_substitute_rank"] = mean(rank_deltas)
    summary["success_rate_primary_metric_gt_0"] = summary[f"success_rate_delta_{selector_metric}_gt_0"]
    summary["mean_delta_primary_metric"] = summary[f"mean_delta_{selector_metric}"]

    per_boost = []
    for boost in boosts:
        matched = []
        for row in rows:
            hit = next((x for x in row["boost_results"] if float(x["sense_boost"]) == float(boost)), None)
            if hit is not None:
                matched.append(hit)
        out = {"sense_boost": float(boost), "num_cases": len(matched)}
        for metric in ["weighted_log_mass", "unweighted_log_mass", "top_substitute_logprob"]:
            deltas = [x[f"delta_{metric}"] for x in matched]
            out[f"mean_delta_{metric}"] = mean(deltas)
            out[f"mean_delta_{metric}_ci95"] = list(bootstrap_mean_ci(deltas))
            out[f"success_rate_delta_{metric}_gt_0"] = mean([1.0 if d > 0 else 0.0 for d in deltas])
        out["mean_kl_base_to_intervened"] = mean([x["kl_base_to_intervened"] for x in matched])
        out["mean_delta_top_substitute_rank"] = mean([x["delta_top_substitute_rank"] for x in matched])
        per_boost.append(out)
    summary["per_boost"] = per_boost

    by_pos = defaultdict(list)
    for row in rows:
        by_pos[row.get("posTT") or row.get("posMASC") or "UNK"].append(row)
    pos_summary = {}
    for pos, pos_rows in sorted(by_pos.items()):
        deltas = [r[f"delta_{selector_metric}"] for r in pos_rows]
        pos_summary[pos] = {
            "num_cases": len(pos_rows),
            "mean_delta_primary_metric": mean(deltas),
            "success_rate_primary_metric_gt_0": mean([1.0 if d > 0 else 0.0 for d in deltas]),
        }
    summary["by_pos"] = pos_summary
    return summary


def main() -> None:
    args = parse_args()
    boosts = parse_boosts(args.boosts)
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    model, tok = load_model_and_tokenizer(args)
    lm_w = get_lm_weight(model).to(args.device)

    cases, filter_summary = choose_cases(payload, args, tok)
    rng = random.Random(args.seed)
    rows = []
    skipped: Counter[str] = Counter()

    for case in tqdm(cases, desc="coinco_lexsub"):
        with torch.no_grad():
            row, reason = score_case(case, model, tok, lm_w, args, boosts, rng)
        if row is None:
            skipped[reason or "unknown"] += 1
            continue
        rows.append(row)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "evals/score_coinco_lexsub_steering.py",
        "method": "coinco_lexsub_sense_steering",
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "split": args.split,
        "selector": args.selector,
        "selector_metric": args.selector_metric,
        "interface": "hidden_coordinate" if args.selector.startswith("hidden_") else "sense",
        "intervention_scope": "hidden_coordinate" if args.selector.startswith("hidden_") else args.intervention_scope,
        "sense_budget_scope": args.intervention_scope if args.selector.startswith("hidden_") else None,
        "boosts": boosts,
        "seed": args.seed,
        "max_cases": args.max_cases,
        "dtype": args.dtype,
        "device": args.device,
        "filters": {
            "exclude_problematic": args.exclude_problematic,
            "one_token_substitutes": args.one_token_substitutes,
            "min_substitute_freq": args.min_substitute_freq,
        },
        "source": payload.get("source", {}),
        "case_file": str(args.cases),
        "case_file_stats": payload.get("stats", {}),
        "filter_summary": filter_summary,
        "skipped_during_scoring": dict(sorted(skipped.items())),
        "summary": summarize_rows(rows, boosts, args.selector_metric),
        "cases": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote CoInCo steering results to {args.out}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
