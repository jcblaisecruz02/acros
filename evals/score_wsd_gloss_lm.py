#!/usr/bin/env python
"""Score WSD by asking a causal LM to choose the most likely WordNet gloss."""

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
    sense_key_lemma,
    sense_key_pos,
)


DEFAULT_MODEL = "jcblaise/sense-smollm2-360M-k32"
POS_TO_DATA_FILE = {
    "n": "data.noun",
    "v": "data.verb",
    "a": "data.adj",
    "r": "data.adv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raganato-root", type=Path, default=Path("data/wsd/WSD_Evaluation_Framework"))
    p.add_argument("--eval", action="append", default=None, help="Repeatable label=xml:gold spec.")
    p.add_argument("--wordnet-dict-dir", type=Path, default=Path("data/wsd/dict"))
    p.add_argument("--wordnet-index-sense", type=Path, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", type=Path, default=Path("eval_logs/wsd/gloss_lm_wsd.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-eval-instances", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--score-mode",
        choices=["conditional", "pmi"],
        default="conditional",
        help="conditional scores p(gloss | sentence,target); pmi subtracts p(gloss | target-only).",
    )
    p.add_argument(
        "--gloss-source",
        choices=["definition", "full"],
        default="definition",
        help="Use only the WordNet definition or the full gloss including examples.",
    )
    p.add_argument("--no-length-normalize", action="store_true")
    p.add_argument("--include-predictions", action="store_true")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


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


def load_wordnet_index(path: Path) -> Dict[str, Tuple[str, str, int]]:
    rows: Dict[str, Tuple[str, str, int]] = {}
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
            rows[sense_key] = (parts[1], pos, sense_number)
    return rows


def load_wordnet_inventory(index_rows: Mapping[str, Tuple[str, str, int]]) -> Dict[str, List[str]]:
    numbered: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for sense_key, (_, pos, sense_number) in index_rows.items():
        key = lemma_pos_key(sense_key_lemma(sense_key), pos)
        numbered[key].append((sense_number, sense_key))
    return {key: [sense for _, sense in sorted(rows)] for key, rows in numbered.items()}


def clean_gloss(gloss: str, source: str) -> str:
    gloss = gloss.strip()
    if source == "definition":
        gloss = gloss.split(";", 1)[0].strip()
    return " ".join(gloss.split())


def load_synset_glosses(dict_dir: Path, gloss_source: str) -> Dict[Tuple[str, str], str]:
    glosses: Dict[Tuple[str, str], str] = {}
    for pos, filename in POS_TO_DATA_FILE.items():
        path = dict_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"WordNet data file not found: {path}")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line or not line[0].isdigit():
                    continue
                before, sep, gloss = line.partition("|")
                if not sep:
                    continue
                parts = before.split()
                if len(parts) < 3:
                    continue
                offset = parts[0]
                ss_type = parts[2]
                synset_pos = "a" if ss_type == "s" else ss_type
                if synset_pos not in POS_TO_DATA_FILE:
                    continue
                gloss_text = clean_gloss(gloss, gloss_source)
                if gloss_text:
                    glosses[(offset, synset_pos)] = gloss_text
    return glosses


def load_sense_glosses(
    index_rows: Mapping[str, Tuple[str, str, int]],
    dict_dir: Path,
    gloss_source: str,
) -> Dict[str, str]:
    synset_glosses = load_synset_glosses(dict_dir, gloss_source)
    sense_glosses: Dict[str, str] = {}
    for sense_key, (offset, pos, _) in index_rows.items():
        gloss = synset_glosses.get((offset, pos))
        if gloss:
            sense_glosses[sense_key] = gloss
    return sense_glosses


def sentence_text(inst: WSDInstance) -> str:
    return " ".join(inst.tokens)


def conditional_prompt(inst: WSDInstance) -> str:
    return (
        f"Sentence: {sentence_text(inst)}\n"
        f'In this sentence, the word "{inst.target_text}" means:'
    )


def prior_prompt(inst: WSDInstance) -> str:
    return f'The word "{inst.target_text}" means:'


def gloss_continuation(gloss: str) -> str:
    gloss = gloss.strip()
    if gloss and gloss[-1] not in ".!?":
        gloss = f"{gloss}."
    return f" {gloss}"


def batched(items: Sequence[Tuple[str, str]], batch_size: int) -> Iterable[Sequence[Tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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


@torch.no_grad()
def score_prompt_continuations(
    model,
    tokenizer,
    pairs: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
    desc: str,
) -> Tuple[List[float | None], List[int]]:
    device = torch.device(args.device)
    scores: List[float | None] = []
    token_counts: List[int] = []

    for batch in tqdm(list(batched(pairs, args.batch_size)), desc=desc):
        prompts = [prompt for prompt, _ in batch]
        continuations = [continuation for _, continuation in batch]
        full_texts = [prompt + continuation for prompt, continuation in batch]

        prompt_enc = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]
        full_enc = tokenizer(
            full_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        full_enc = {key: value.to(device) for key, value in full_enc.items()}
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        out = model(**full_enc, use_cache=False, return_dict=True)
        log_probs = out.logits[:, :-1, :].float().log_softmax(dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        shifted_attention = attention_mask[:, 1:].bool()

        positions = torch.arange(input_ids.shape[1], device=device)
        for row_idx, prompt_len in enumerate(prompt_lens):
            full_len = int(attention_mask[row_idx].sum().item())
            token_positions = positions[1:full_len]
            continuation_mask = token_positions >= prompt_len
            valid_mask = shifted_attention[row_idx, : full_len - 1] & continuation_mask
            count = int(valid_mask.sum().item())
            if count == 0:
                scores.append(None)
                token_counts.append(0)
                continue
            total = token_log_probs[row_idx, : full_len - 1][valid_mask].sum().item()
            score = total if args.no_length_normalize else total / count
            scores.append(score)
            token_counts.append(count)

    return scores, token_counts


def build_candidate_records(
    instances: Sequence[WSDInstance],
    inventory: Mapping[str, Sequence[str]],
    glosses: Mapping[str, str],
) -> Tuple[List[Dict], Counter]:
    records: List[Dict] = []
    stats: Counter = Counter()
    for inst_idx, inst in enumerate(instances):
        candidates = list(inventory.get(lemma_pos_key(inst.lemma, inst.pos), ()))
        if not candidates:
            stats["missing_inventory"] += 1
            continue
        stats["candidate_total"] += len(candidates)
        for sense_key in candidates:
            gloss = glosses.get(sense_key)
            if not gloss:
                stats["missing_candidate_gloss"] += 1
                continue
            records.append(
                {
                    "inst_idx": inst_idx,
                    "sense_key": sense_key,
                    "gloss": gloss,
                }
            )
            stats["scored_candidate_total"] += 1
    return records, stats


def score_instances(
    instances: Sequence[WSDInstance],
    records: Sequence[Mapping],
    conditional_scores: Sequence[float | None],
    prior_scores: Sequence[float | None] | None,
    token_counts: Sequence[int],
    first_sense: Mapping[str, str],
    include_predictions: bool,
) -> Dict:
    grouped: Dict[int, List[Tuple[str, float, str, int]]] = defaultdict(list)
    skipped_candidate_scores = 0

    if prior_scores is None:
        for record, cond_score, token_count in zip(records, conditional_scores, token_counts):
            if cond_score is None:
                skipped_candidate_scores += 1
                continue
            grouped[record["inst_idx"]].append((record["sense_key"], cond_score, record["gloss"], token_count))
    else:
        for record, cond_score, prior_score, token_count in zip(records, conditional_scores, prior_scores, token_counts):
            if cond_score is None or prior_score is None:
                skipped_candidate_scores += 1
                continue
            grouped[record["inst_idx"]].append(
                (record["sense_key"], cond_score - prior_score, record["gloss"], token_count)
            )

    total = answered = correct = missing_gold = missing_prediction = 0
    lm_predictions = first_sense_fallback = no_scored_candidates = 0
    candidate_counts: List[int] = []
    dataset_counts: Dict[str, Counter] = defaultdict(Counter)
    predictions: List[Dict] = []

    for idx, inst in enumerate(instances):
        if not inst.gold_sense_keys:
            missing_gold += 1
            continue
        total += 1
        candidates = grouped.get(idx, [])
        if candidates:
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
            pred, score, gloss, token_count = candidates[0]
            source = "gloss_lm"
            lm_predictions += 1
            candidate_counts.append(len(candidates))
        else:
            pred = first_sense.get(lemma_pos_key(inst.lemma, inst.pos))
            score = None
            gloss = ""
            token_count = 0
            source = "wordnet_first_sense" if pred else "missing"
            first_sense_fallback += int(pred is not None)
            no_scored_candidates += 1

        is_correct = pred in inst.gold_sense_keys if pred else False
        if pred is None:
            missing_prediction += 1
        else:
            answered += 1
            correct += int(is_correct)

        bucket = dataset_counts[inst.dataset]
        bucket["total"] += 1
        bucket["answered"] += int(pred is not None)
        bucket["correct"] += int(is_correct)
        bucket["lm_predictions"] += int(source == "gloss_lm")
        bucket["first_sense_fallback"] += int(source == "wordnet_first_sense")
        bucket["missing_prediction"] += int(pred is None)

        if include_predictions:
            predictions.append(
                {
                    "dataset": inst.dataset,
                    "instance_id": inst.instance_id,
                    "sentence": sentence_text(inst),
                    "target_index": inst.target_index,
                    "target_text": inst.target_text,
                    "lemma": inst.lemma,
                    "pos": inst.pos,
                    "gold_sense_keys": list(inst.gold_sense_keys),
                    "prediction": pred,
                    "source": source,
                    "score": score,
                    "gloss": gloss,
                    "token_count": token_count,
                    "num_scored_candidates": len(candidates),
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
            "lm_predictions": counts["lm_predictions"],
            "first_sense_fallback": counts["first_sense_fallback"],
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
            "missing_prediction": missing_prediction,
            "lm_predictions": lm_predictions,
            "first_sense_fallback": first_sense_fallback,
            "no_scored_candidates": no_scored_candidates,
            "skipped_candidate_scores": skipped_candidate_scores,
            "mean_scored_candidates": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        },
        "by_dataset": by_dataset,
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def main() -> None:
    args = parse_args()
    index_sense = args.wordnet_index_sense or (args.wordnet_dict_dir / "index.sense")
    eval_specs = prepare_eval_specs(args)
    instances = truncate(load_instances(eval_specs, kind="eval"), args.max_eval_instances)
    index_rows = load_wordnet_index(index_sense)
    inventory = load_wordnet_inventory(index_rows)
    first_sense = load_wordnet_first_sense(index_sense)
    glosses = load_sense_glosses(index_rows, args.wordnet_dict_dir, args.gloss_source)
    records, candidate_stats = build_candidate_records(instances, inventory, glosses)

    model, tokenizer = load_model_and_tokenizer(args)
    conditional_pairs = [
        (conditional_prompt(instances[record["inst_idx"]]), gloss_continuation(record["gloss"]))
        for record in records
    ]
    conditional_scores, token_counts = score_prompt_continuations(
        model,
        tokenizer,
        conditional_pairs,
        args,
        desc="Scoring conditional glosses",
    )

    prior_scores = None
    if args.score_mode == "pmi":
        prior_pairs = [
            (prior_prompt(instances[record["inst_idx"]]), gloss_continuation(record["gloss"]))
            for record in records
        ]
        prior_scores, _ = score_prompt_continuations(
            model,
            tokenizer,
            prior_pairs,
            args,
            desc="Scoring gloss priors",
        )

    scores = score_instances(
        instances,
        records,
        conditional_scores,
        prior_scores,
        token_counts,
        first_sense,
        include_predictions=args.include_predictions,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "wordnet_gloss_lm_scoring",
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "eval_specs": eval_specs,
        "wordnet_dict_dir": str(args.wordnet_dict_dir),
        "wordnet_index_sense": str(index_sense),
        "score_mode": args.score_mode,
        "gloss_source": args.gloss_source,
        "length_normalize": not args.no_length_normalize,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "num_eval_instances_loaded": len(instances),
        "num_sense_glosses": len(glosses),
        "candidate_stats": dict(candidate_stats),
        "scores": scores,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    overall = scores["overall"]
    print(
        "Gloss-LM WSD: "
        f"F1={overall['f1']:.4f} "
        f"accuracy={overall['accuracy']:.4f} "
        f"answered={overall['answered']}/{overall['total']} "
        f"lm_predictions={overall['lm_predictions']} "
        f"first_sense_fallback={overall['first_sense_fallback']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
