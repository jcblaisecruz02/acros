#!/usr/bin/env python
"""Score WSD by matching context and WordNet gloss sense activations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - only used in lean smoke environments.
    def tqdm(iterable, **kwargs):
        return iterable

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from score_wsd_centroid import (  # noqa: E402
    DEFAULT_MODEL,
    compute_activation,
    find_token_positions,
    load_model_and_tokenizer,
    text_and_spans,
)
from score_wsd_gloss_lm import (  # noqa: E402
    load_sense_glosses,
    load_wordnet_index,
    load_wordnet_inventory,
)
from score_wsd_mfs import (  # noqa: E402
    WSDInstance,
    autodiscover_raganato,
    lemma_pos_key,
    load_instances,
    load_wordnet_first_sense,
    normalize_lemma,
)


@dataclass(frozen=True)
class ActivationRequest:
    key: str
    text: str
    target_span: Tuple[int, int] | None = None
    target_strategy: str = "span"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raganato-root", type=Path, default=Path("data/wsd/WSD_Evaluation_Framework"))
    p.add_argument("--eval", action="append", default=None, help="Repeatable label=xml:gold spec.")
    p.add_argument("--wordnet-dict-dir", type=Path, default=Path("data/wsd/dict"))
    p.add_argument("--wordnet-index-sense", type=Path, default=Path("data/wsd/dict/index.sense"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", type=Path, default=Path("eval_logs/wsd/gloss_activation_wsd.json"))
    p.add_argument("--activation-cache", type=Path, default=None)
    p.add_argument("--overwrite-activation-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--max-eval-instances", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--activation-mode",
        choices=["target_contribution_norm", "future_target_attention", "target_plus_future"],
        default="future_target_attention",
    )
    p.add_argument(
        "--probe-format",
        choices=["lemma_colon_gloss", "the_word_means", "lemma_means", "bare_gloss_last"],
        default="lemma_colon_gloss",
    )
    p.add_argument(
        "--gloss-source",
        choices=["definition", "full"],
        default="definition",
        help="Use only the WordNet definition or the full gloss including examples.",
    )
    p.add_argument("--normalize-activations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-predictions", action="store_true")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def prepare_eval_specs(args: argparse.Namespace) -> List[str]:
    eval_specs = list(args.eval or [])
    if args.raganato_root and not eval_specs:
        _, auto_eval = autodiscover_raganato(args.raganato_root)
        eval_specs = auto_eval
    if not eval_specs:
        raise ValueError("No eval data supplied. Use --eval or --raganato-root.")
    return eval_specs


def truncate(instances: List[WSDInstance], limit: int) -> List[WSDInstance]:
    return instances[:limit] if limit and limit > 0 else instances


def activation_key(inst: WSDInstance) -> str:
    return f"{inst.dataset}::{inst.instance_id}"


def lemma_display(lemma: str) -> str:
    return normalize_lemma(lemma).replace("_", " ")


def build_context_request(inst: WSDInstance) -> ActivationRequest | None:
    text, spans = text_and_spans(inst.tokens)
    if inst.target_index >= len(spans):
        return None
    return ActivationRequest(
        key=activation_key(inst),
        text=text,
        target_span=spans[inst.target_index],
        target_strategy="span",
    )


def build_gloss_request(sense_key: str, lemma: str, gloss: str, probe_format: str) -> ActivationRequest:
    lemma_text = lemma_display(lemma)
    gloss = " ".join(gloss.strip().split())

    if probe_format == "lemma_colon_gloss":
        text = f"{lemma_text}: {gloss}"
        return ActivationRequest(sense_key, text, (0, len(lemma_text)), "span")
    if probe_format == "the_word_means":
        prefix = "The word "
        text = f"{prefix}{lemma_text} means: {gloss}"
        start = len(prefix)
        return ActivationRequest(sense_key, text, (start, start + len(lemma_text)), "span")
    if probe_format == "lemma_means":
        text = f"{lemma_text} means: {gloss}"
        return ActivationRequest(sense_key, text, (0, len(lemma_text)), "span")
    if probe_format == "bare_gloss_last":
        return ActivationRequest(sense_key, gloss, None, "last")
    raise ValueError(f"Unknown probe format: {probe_format}")


def batched(items: Sequence[ActivationRequest], batch_size: int) -> Iterable[Sequence[ActivationRequest]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def token_positions_for_request(
    req: ActivationRequest,
    offsets: Sequence[Tuple[int, int]],
    seq_len: int,
) -> List[int]:
    if req.target_strategy == "last":
        return [max(0, seq_len - 1)]
    if req.target_strategy != "span" or req.target_span is None:
        raise ValueError(f"Unsupported target strategy for request {req.key!r}: {req.target_strategy}")
    return find_token_positions(offsets, req.target_span)


@torch.no_grad()
def extract_request_activations(
    model,
    tokenizer,
    requests: Sequence[ActivationRequest],
    args: argparse.Namespace,
    desc: str,
) -> Tuple[Dict[str, torch.Tensor], Counter]:
    device = torch.device(args.device)
    activations: Dict[str, torch.Tensor] = {}
    stats: Counter = Counter()

    for batch in tqdm(list(batched(requests, args.batch_size)), desc=desc):
        texts = [req.text for req in batch]
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

        for row_idx, req in enumerate(batch):
            seq_len = int(enc["attention_mask"][row_idx].sum().item())
            positions = token_positions_for_request(req, offsets[row_idx], seq_len)
            if not positions:
                stats["missing_token_alignment"] += 1
                continue
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
            activations[req.key] = vec
            stats["extracted"] += 1

    return activations, stats


def load_or_extract_gloss_activations(
    model,
    tokenizer,
    requests: Sequence[ActivationRequest],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    if args.activation_cache and args.activation_cache.exists() and not args.overwrite_activation_cache:
        payload = torch.load(args.activation_cache, map_location="cpu")
        for field, requested in (
            ("model", args.model),
            ("activation_mode", args.activation_mode),
            ("probe_format", args.probe_format),
            ("gloss_source", args.gloss_source),
            ("normalize_activations", args.normalize_activations),
        ):
            cached = payload.get(field)
            if cached != requested:
                raise ValueError(
                    f"Activation cache {args.activation_cache} has {field}={cached!r}, "
                    f"but this run requested {requested!r}. Use a matching cache or "
                    "--overwrite-activation-cache."
                )
        return payload["activations"], {"loaded_from_cache": str(args.activation_cache)}

    activations, stats = extract_request_activations(
        model,
        tokenizer,
        requests,
        args,
        desc="Extracting gloss activations",
    )
    cache_info = {"loaded_from_cache": None, "gloss_activation_stats": dict(stats)}
    if args.activation_cache:
        args.activation_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "activation_mode": args.activation_mode,
                "probe_format": args.probe_format,
                "gloss_source": args.gloss_source,
                "normalize_activations": args.normalize_activations,
                "activations": activations,
            },
            args.activation_cache,
        )
        cache_info["saved_to_cache"] = str(args.activation_cache)
    return activations, cache_info


def predict_by_gloss_activation(
    inst: WSDInstance,
    ctx_vec: torch.Tensor | None,
    inventory: Mapping[str, Sequence[str]],
    sense_glosses: Mapping[str, str],
    gloss_activations: Mapping[str, torch.Tensor],
    first_sense: Mapping[str, str],
) -> Tuple[str | None, str, int]:
    if ctx_vec is None:
        return first_sense.get(lemma_pos_key(inst.lemma, inst.pos)), "wordnet_first_sense_missing_context", 0

    key = lemma_pos_key(inst.lemma, inst.pos)
    candidates = [sense for sense in inventory.get(key, ()) if sense in sense_glosses]
    scored = []
    for sense_key in candidates:
        gloss_vec = gloss_activations.get(sense_key)
        if gloss_vec is None:
            continue
        scored.append((sense_key, torch.dot(ctx_vec, gloss_vec).item()))
    if scored:
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0], "gloss_activation", len(scored)
    return first_sense.get(key), "wordnet_first_sense_missing_gloss_activation", 0


def score_predictions(
    eval_instances: Sequence[WSDInstance],
    ctx_activations: Mapping[str, torch.Tensor],
    inventory: Mapping[str, Sequence[str]],
    sense_glosses: Mapping[str, str],
    gloss_activations: Mapping[str, torch.Tensor],
    first_sense: Mapping[str, str],
    include_predictions: bool,
) -> Dict:
    total = answered = correct = missing_gold = 0
    missing_context = missing_prediction = 0
    gloss_predictions = first_sense_fallback = 0
    candidate_counts: List[int] = []
    dataset_counts: Dict[str, Counter] = defaultdict(Counter)
    predictions: List[Dict] = []

    for inst in eval_instances:
        if not inst.gold_sense_keys:
            missing_gold += 1
            continue
        total += 1
        ctx_vec = ctx_activations.get(activation_key(inst))
        pred, source, num_scored = predict_by_gloss_activation(
            inst,
            ctx_vec,
            inventory,
            sense_glosses,
            gloss_activations,
            first_sense,
        )
        is_correct = pred in inst.gold_sense_keys if pred else False

        if pred is None:
            missing_prediction += 1
        else:
            answered += 1
            correct += int(is_correct)
            gloss_predictions += int(source == "gloss_activation")
            first_sense_fallback += int(source.startswith("wordnet_first_sense"))
            candidate_counts.append(num_scored)

        bucket = dataset_counts[inst.dataset]
        bucket["total"] += 1
        bucket["answered"] += int(pred is not None)
        bucket["correct"] += int(is_correct)
        bucket["gloss_predictions"] += int(source == "gloss_activation")
        bucket["first_sense_fallback"] += int(source.startswith("wordnet_first_sense"))
        bucket["missing_context"] += int(ctx_vec is None)
        bucket["missing_prediction"] += int(pred is None)
        missing_context += int(ctx_vec is None)

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
                    "num_scored_glosses": num_scored,
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
            "gloss_predictions": counts["gloss_predictions"],
            "first_sense_fallback": counts["first_sense_fallback"],
            "missing_context": counts["missing_context"],
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
            "missing_context": missing_context,
            "missing_prediction": missing_prediction,
            "gloss_predictions": gloss_predictions,
            "first_sense_fallback": first_sense_fallback,
            "mean_scored_glosses": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        },
        "by_dataset": by_dataset,
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def main() -> None:
    args = parse_args()
    eval_specs = prepare_eval_specs(args)
    eval_instances = truncate(load_instances(eval_specs, kind="eval"), args.max_eval_instances)

    index_rows = load_wordnet_index(args.wordnet_index_sense)
    inventory = load_wordnet_inventory(index_rows)
    sense_glosses = load_sense_glosses(index_rows, args.wordnet_dict_dir, args.gloss_source)
    first_sense = load_wordnet_first_sense(args.wordnet_index_sense)

    context_requests = [req for inst in eval_instances if (req := build_context_request(inst)) is not None]
    needed_sense_keys = sorted(
        {
            sense_key
            for inst in eval_instances
            for sense_key in inventory.get(lemma_pos_key(inst.lemma, inst.pos), ())
            if sense_key in sense_glosses
        }
    )
    gloss_requests = [
        build_gloss_request(
            sense_key,
            lemma=sense_key.split("%", 1)[0],
            gloss=sense_glosses[sense_key],
            probe_format=args.probe_format,
        )
        for sense_key in needed_sense_keys
    ]

    model, tokenizer = load_model_and_tokenizer(args)
    context_activations, context_stats = extract_request_activations(
        model,
        tokenizer,
        context_requests,
        args,
        desc="Extracting context activations",
    )
    gloss_activations, cache_info = load_or_extract_gloss_activations(
        model,
        tokenizer,
        gloss_requests,
        args,
    )

    scores = score_predictions(
        eval_instances,
        context_activations,
        inventory,
        sense_glosses,
        gloss_activations,
        first_sense,
        include_predictions=args.include_predictions,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": f"wordnet_gloss_activation_matching_{args.activation_mode}",
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "eval_specs": eval_specs,
        "wordnet_dict_dir": str(args.wordnet_dict_dir),
        "wordnet_index_sense": str(args.wordnet_index_sense),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "activation_mode": args.activation_mode,
        "probe_format": args.probe_format,
        "gloss_source": args.gloss_source,
        "normalize_activations": args.normalize_activations,
        "num_eval_instances_loaded": len(eval_instances),
        "num_context_requests": len(context_requests),
        "num_gloss_requests": len(gloss_requests),
        "context_activation_stats": dict(context_stats),
        "cache": cache_info,
        "scores": scores,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    overall = scores["overall"]
    print(
        "Gloss-activation WSD: "
        f"F1={overall['f1']:.4f} "
        f"accuracy={overall['accuracy']:.4f} "
        f"answered={overall['answered']}/{overall['total']} "
        f"gloss_predictions={overall['gloss_predictions']} "
        f"first_sense_fallback={overall['first_sense_fallback']} "
        f"missing_context={overall['missing_context']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
