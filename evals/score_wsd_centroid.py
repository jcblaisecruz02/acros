#!/usr/bin/env python
"""Score WSD with nearest WordNet-sense centroids over induced sense activations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - only used in lean smoke environments.
    def tqdm(iterable, **kwargs):
        return iterable

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from score_wsd_mfs import (  # noqa: E402
    WSDInstance,
    autodiscover_raganato,
    lemma_pos_key,
    load_instances,
    load_wordnet_first_sense,
    normalize_lemma,
    normalize_pos,
    sense_key_lemma,
    sense_key_pos,
)


DEFAULT_MODEL = "jcblaise/sense-smollm2-360M-k32"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raganato-root", type=Path, default=Path("data/wsd/WSD_Evaluation_Framework"))
    p.add_argument("--train", action="append", default=None, help="Repeatable label=xml:gold spec.")
    p.add_argument("--eval", action="append", default=None, help="Repeatable label=xml:gold spec.")
    p.add_argument("--wordnet-index-sense", type=Path, default=Path("data/wsd/dict/index.sense"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", type=Path, default=Path("eval_logs/wsd/sense_k32_centroid_raganato_all.json"))
    p.add_argument("--centroid-cache", type=Path, default=None)
    p.add_argument("--overwrite-centroid-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--max-train-instances", type=int, default=0)
    p.add_argument("--max-eval-instances", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--activation-mode",
        choices=["target_contribution_norm", "future_target_attention", "target_plus_future"],
        default="target_contribution_norm",
        help=(
            "Sense feature to centroid. target_contribution_norm is the original target-position "
            "sense contribution magnitude. future_target_attention measures how much later tokens "
            "route through the target token for each sense. target_plus_future concatenates both."
        ),
    )
    p.add_argument("--normalize-activations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-predictions", action="store_true")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def load_wordnet_inventory(path: Path) -> Dict[str, List[str]]:
    numbered: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected index.sense row")
            sense_key = parts[0]
            pos = sense_key_pos(sense_key)
            if pos is None:
                continue
            try:
                sense_number = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid sense_number {parts[2]!r}") from exc
            key = lemma_pos_key(sense_key_lemma(sense_key), pos)
            numbered[key].append((sense_number, sense_key))
    return {key: [sense for _, sense in sorted(rows)] for key, rows in numbered.items()}


def text_and_spans(tokens: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    text_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return "".join(text_parts), spans


def find_token_positions(offsets: Sequence[Tuple[int, int]], span: Tuple[int, int]) -> List[int]:
    start, end = span
    positions: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if not (tok_end <= start or tok_start >= end):
            positions.append(idx)
    return positions


def batched(items: Sequence[WSDInstance], batch_size: int) -> Iterable[Sequence[WSDInstance]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def activation_key(inst: WSDInstance) -> str:
    return f"{inst.dataset}::{inst.instance_id}"


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=args.trust_remote_code,
        dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(torch.device(args.device))
    return model, tokenizer


def target_contribution_norm(
    contextualization: torch.Tensor,
    senses: torch.Tensor,
    target_positions: Sequence[int],
) -> torch.Tensor:
    ctx_i = contextualization[:, target_positions, :].float()
    senses_i = senses.float()
    # Shape: [num_senses, target_subtokens, hidden_size].
    per_sense_target_mix = torch.matmul(ctx_i, senses_i)
    return per_sense_target_mix.norm(dim=-1).mean(dim=-1)


def future_target_attention(
    contextualization: torch.Tensor,
    target_positions: Sequence[int],
    seq_len: int,
    stats: Counter,
) -> torch.Tensor:
    target_idx = torch.tensor(target_positions, device=contextualization.device, dtype=torch.long)
    first_target = min(target_positions)
    last_target = max(target_positions)

    if last_target + 1 < seq_len:
        query_idx = torch.arange(last_target + 1, seq_len, device=contextualization.device)
        stats["future_query_used"] += 1
    else:
        # Sentence-final targets have no right-context queries. Fall back to the
        # target query itself instead of dropping the instance.
        query_idx = torch.arange(first_target, last_target + 1, device=contextualization.device)
        stats["target_query_fallback"] += 1

    # contextualization[k, query, source] is normalized over source positions for
    # each query. This keeps only the mass assigned to the target token as source.
    target_mass = contextualization[:, query_idx, :][:, :, target_idx].float()
    return target_mass.mean(dim=(1, 2))


def compute_activation(
    contextualization: torch.Tensor,
    senses: torch.Tensor,
    target_positions: Sequence[int],
    seq_len: int,
    args: argparse.Namespace,
    stats: Counter,
) -> torch.Tensor:
    if args.activation_mode == "target_contribution_norm":
        return target_contribution_norm(contextualization, senses, target_positions)
    if args.activation_mode == "future_target_attention":
        return future_target_attention(contextualization, target_positions, seq_len, stats)
    if args.activation_mode == "target_plus_future":
        target_vec = target_contribution_norm(contextualization, senses, target_positions)
        future_vec = future_target_attention(contextualization, target_positions, seq_len, stats)
        return torch.cat([target_vec, future_vec], dim=0)
    raise ValueError(f"Unknown activation mode: {args.activation_mode}")


@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    instances: Sequence[WSDInstance],
    args: argparse.Namespace,
    desc: str,
) -> Tuple[Dict[str, torch.Tensor], Counter]:
    device = torch.device(args.device)
    activations: Dict[str, torch.Tensor] = {}
    stats: Counter = Counter()

    for batch in tqdm(list(batched(instances, args.batch_size)), desc=desc):
        texts: List[str] = []
        target_spans: List[Tuple[int, int]] = []
        kept_instances: List[WSDInstance] = []
        for inst in batch:
            text, spans = text_and_spans(inst.tokens)
            if inst.target_index >= len(spans):
                stats["bad_target_index"] += 1
                continue
            texts.append(text)
            target_spans.append(spans[inst.target_index])
            kept_instances.append(inst)

        if not kept_instances:
            continue

        enc = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        offsets = enc.pop("offset_mapping").tolist()
        enc = {key: value.to(device) for key, value in enc.items()}

        out = model(**enc, use_cache=False, return_dict=True)
        contextualization = getattr(out, "contextualization", None)
        senses = getattr(out, "senses", None)
        if contextualization is None or senses is None:
            raise RuntimeError("Model output does not expose contextualization and senses.")

        contextualization = contextualization.detach()
        senses = senses.detach()

        for row_idx, inst in enumerate(kept_instances):
            positions = find_token_positions(offsets[row_idx], target_spans[row_idx])
            if not positions:
                stats["missing_token_alignment"] += 1
                continue
            seq_len = int(enc["attention_mask"][row_idx].sum().item())
            vec = compute_activation(
                contextualization[row_idx],
                senses[row_idx],
                positions,
                seq_len,
                args,
                stats,
            ).cpu()
            if args.normalize_activations:
                vec = vec / vec.norm(p=2).clamp_min(1e-12)
            activations[activation_key(inst)] = vec
            stats["extracted"] += 1

    return activations, stats


def build_centroids(
    train_instances: Sequence[WSDInstance],
    train_activations: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = Counter()
    for inst in train_instances:
        vec = train_activations.get(activation_key(inst))
        if vec is None:
            continue
        for sense_key in inst.gold_sense_keys:
            if sense_key not in sums:
                sums[sense_key] = torch.zeros_like(vec)
            sums[sense_key] += vec
            counts[sense_key] += 1

    centroids: Dict[str, torch.Tensor] = {}
    for sense_key, total in sums.items():
        centroid = total / max(counts[sense_key], 1)
        if args.normalize_activations:
            centroid = centroid / centroid.norm(p=2).clamp_min(1e-12)
        centroids[sense_key] = centroid
    return centroids, dict(counts)


def load_or_build_centroids(
    model,
    tokenizer,
    train_instances: Sequence[WSDInstance],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict]:
    if args.centroid_cache and args.centroid_cache.exists() and not args.overwrite_centroid_cache:
        payload = torch.load(args.centroid_cache, map_location="cpu")
        cache_activation_mode = payload.get("activation_mode", "target_contribution_norm")
        if cache_activation_mode != args.activation_mode:
            raise ValueError(
                f"Centroid cache {args.centroid_cache} was built with activation_mode="
                f"{cache_activation_mode!r}, but this run requested {args.activation_mode!r}. "
                "Use a matching cache path or --overwrite-centroid-cache."
            )
        return payload["centroids"], payload["counts"], {"loaded_from_cache": str(args.centroid_cache)}

    train_activations, extract_stats = extract_activations(
        model,
        tokenizer,
        train_instances,
        args,
        desc="Extracting SemCor activations",
    )
    centroids, counts = build_centroids(train_instances, train_activations, args)
    cache_info = {
        "loaded_from_cache": None,
        "train_activation_stats": dict(extract_stats),
    }
    if args.centroid_cache:
        args.centroid_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "activation_mode": args.activation_mode,
                "normalize_activations": args.normalize_activations,
                "centroids": centroids,
                "counts": counts,
            },
            args.centroid_cache,
        )
        cache_info["saved_to_cache"] = str(args.centroid_cache)
    return centroids, counts, cache_info


def nearest_centroid_prediction(
    inst: WSDInstance,
    vec: torch.Tensor,
    inventory: Mapping[str, Sequence[str]],
    centroids: Mapping[str, torch.Tensor],
    first_sense: Mapping[str, str],
) -> Tuple[str | None, str, int]:
    key = lemma_pos_key(inst.lemma, inst.pos)
    candidates = list(inventory.get(key, ()))
    scored = [(sense_key, torch.dot(vec, centroids[sense_key]).item()) for sense_key in candidates if sense_key in centroids]
    if scored:
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0], "centroid", len(scored)
    return first_sense.get(key), "wordnet_first_sense", 0


def score_centroids(
    eval_instances: Sequence[WSDInstance],
    eval_activations: Mapping[str, torch.Tensor],
    inventory: Mapping[str, Sequence[str]],
    centroids: Mapping[str, torch.Tensor],
    first_sense: Mapping[str, str],
    include_predictions: bool,
) -> Dict:
    total = answered = correct = missing_gold = missing_activation = missing_prediction = 0
    centroid_predictions = first_sense_fallback = 0
    candidate_centroid_counts: List[int] = []
    dataset_counts: Dict[str, Counter] = defaultdict(Counter)
    predictions: List[Dict] = []

    for inst in eval_instances:
        if not inst.gold_sense_keys:
            missing_gold += 1
            continue
        total += 1
        vec = eval_activations.get(activation_key(inst))
        if vec is None:
            missing_activation += 1
            pred = first_sense.get(lemma_pos_key(inst.lemma, inst.pos))
            source = "wordnet_first_sense_missing_activation" if pred else "missing"
            num_candidate_centroids = 0
        else:
            pred, source, num_candidate_centroids = nearest_centroid_prediction(
                inst,
                vec,
                inventory,
                centroids,
                first_sense,
            )
        is_correct = pred in inst.gold_sense_keys if pred else False

        if pred is None:
            missing_prediction += 1
        else:
            answered += 1
            correct += int(is_correct)
            centroid_predictions += int(source == "centroid")
            first_sense_fallback += int(source.startswith("wordnet_first_sense"))
            candidate_centroid_counts.append(num_candidate_centroids)

        bucket = dataset_counts[inst.dataset]
        bucket["total"] += 1
        bucket["answered"] += int(pred is not None)
        bucket["correct"] += int(is_correct)
        bucket["centroid_predictions"] += int(source == "centroid")
        bucket["first_sense_fallback"] += int(source.startswith("wordnet_first_sense"))
        bucket["missing_activation"] += int(vec is None)
        bucket["missing_prediction"] += int(pred is None)

        if include_predictions:
            predictions.append(
                {
                    "dataset": inst.dataset,
                    "instance_id": inst.instance_id,
                    "sentence": " ".join(inst.tokens),
                    "target_index": inst.target_index,
                    "target_text": inst.target_text,
                    "lemma": inst.lemma,
                    "pos": inst.pos,
                    "gold_sense_keys": list(inst.gold_sense_keys),
                    "prediction": pred,
                    "source": source,
                    "num_candidate_centroids": num_candidate_centroids,
                    "correct": is_correct,
                }
            )

    precision = correct / answered if answered else 0.0
    recall = correct / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_dataset = {}
    for dataset, counts in sorted(dataset_counts.items()):
        d_total = counts["total"]
        d_answered = counts["answered"]
        d_correct = counts["correct"]
        d_precision = d_correct / d_answered if d_answered else 0.0
        d_recall = d_correct / d_total if d_total else 0.0
        d_f1 = 2 * d_precision * d_recall / (d_precision + d_recall) if d_precision + d_recall else 0.0
        by_dataset[dataset] = {
            "total": d_total,
            "answered": d_answered,
            "correct": d_correct,
            "precision": d_precision,
            "recall": d_recall,
            "f1": d_f1,
            "accuracy": d_correct / d_total if d_total else 0.0,
            "centroid_predictions": counts["centroid_predictions"],
            "first_sense_fallback": counts["first_sense_fallback"],
            "missing_activation": counts["missing_activation"],
            "missing_prediction": counts["missing_prediction"],
        }

    result = {
        "overall": {
            "total": total,
            "answered": answered,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": correct / total if total else 0.0,
            "missing_gold": missing_gold,
            "missing_activation": missing_activation,
            "missing_prediction": missing_prediction,
            "centroid_predictions": centroid_predictions,
            "first_sense_fallback": first_sense_fallback,
            "mean_candidate_centroids": (
                sum(candidate_centroid_counts) / len(candidate_centroid_counts) if candidate_centroid_counts else 0.0
            ),
        },
        "by_dataset": by_dataset,
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def prepare_specs(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    train_specs = list(args.train or [])
    eval_specs = list(args.eval or [])
    if args.raganato_root:
        auto_train, auto_eval = autodiscover_raganato(args.raganato_root)
        train_specs = train_specs or auto_train
        eval_specs = eval_specs or auto_eval
    if not train_specs:
        raise ValueError("No training data supplied. Use --train or --raganato-root.")
    if not eval_specs:
        raise ValueError("No eval data supplied. Use --eval or --raganato-root.")
    return train_specs, eval_specs


def truncate(instances: List[WSDInstance], limit: int) -> List[WSDInstance]:
    return instances[:limit] if limit and limit > 0 else instances


def main() -> None:
    args = parse_args()
    train_specs, eval_specs = prepare_specs(args)
    train_instances = truncate(load_instances(train_specs, kind="train"), args.max_train_instances)
    eval_instances = truncate(load_instances(eval_specs, kind="eval"), args.max_eval_instances)
    inventory = load_wordnet_inventory(args.wordnet_index_sense)
    first_sense = load_wordnet_first_sense(args.wordnet_index_sense)

    model, tokenizer = load_model_and_tokenizer(args)
    centroids, centroid_counts, cache_info = load_or_build_centroids(model, tokenizer, train_instances, args)
    eval_activations, eval_extract_stats = extract_activations(
        model,
        tokenizer,
        eval_instances,
        args,
        desc="Extracting eval activations",
    )

    scores = score_centroids(
        eval_instances,
        eval_activations,
        inventory,
        centroids,
        first_sense,
        include_predictions=args.include_predictions,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": f"nearest_wordnet_sense_centroid_over_{args.activation_mode}",
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "train_specs": train_specs,
        "eval_specs": eval_specs,
        "wordnet_index_sense": str(args.wordnet_index_sense),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "activation_mode": args.activation_mode,
        "normalize_activations": args.normalize_activations,
        "num_train_instances_loaded": len(train_instances),
        "num_eval_instances_loaded": len(eval_instances),
        "num_centroids": len(centroids),
        "centroid_count_summary": {
            "min": min(centroid_counts.values()) if centroid_counts else 0,
            "max": max(centroid_counts.values()) if centroid_counts else 0,
            "mean": sum(centroid_counts.values()) / len(centroid_counts) if centroid_counts else 0.0,
        },
        "cache": cache_info,
        "eval_activation_stats": dict(eval_extract_stats),
        "scores": scores,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    overall = scores["overall"]
    print(
        "Centroid WSD: "
        f"F1={overall['f1']:.4f} "
        f"accuracy={overall['accuracy']:.4f} "
        f"answered={overall['answered']}/{overall['total']} "
        f"centroid_predictions={overall['centroid_predictions']} "
        f"first_sense_fallback={overall['first_sense_fallback']} "
        f"missing_activation={overall['missing_activation']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
