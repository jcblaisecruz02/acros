#!/usr/bin/env python
"""Run a small, auditable XL-Sum generation evaluation.

This script intentionally keeps the protocol simple:

1. deterministic XL-Sum test subset selection,
2. deterministic greedy decoding,
3. per-example JSONL outputs,
4. aggregate JSON with lexical overlap and generation-health metrics.

It is not a replacement for a full summarization leaderboard run; it is a
paper-facing fluency/downstream probe for the current ACROS comparison set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LANG_TO_XLSUM = {
    "ind": "indonesian",
    "est": "estonian",
    "swh": "swahili",
    "tur": "turkish",
}

PROMPTS = {
    "ind": ("Artikel:\n", "\n\nRingkasan:"),
    "swh": ("Makala:\n", "\n\nMuhtasari:"),
    "tur": ("Haber:\n", "\n\nÖzet:"),
}

INSTRUCTION_PREFIXES = {
    "ind": "Ringkas artikel berikut dalam satu kalimat bahasa Indonesia.\n\nArtikel:\n",
    "swh": "Fupisha makala ifuatayo kwa sentensi moja ya Kiswahili.\n\nMakala:\n",
    "tur": "Aşağıdaki haberi Türkçe tek cümleyle özetle.\n\nHaber:\n",
}

SUMMARY_LABELS = {
    "ind": ("Ringkasan:", "Ringkasan"),
    "swh": ("Muhtasari:", "Muhtasari"),
    "tur": ("Özet:", "Özet", "Ozet:", "Ozet"),
}

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score XL-Sum summaries.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_key", required=True, help="Stable artifact key, e.g. acros_k32_ind")
    parser.add_argument("--model_label", required=True, help="Human-readable table label")
    parser.add_argument("--lang", required=True, choices=sorted(LANG_TO_XLSUM))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=128, help="Deterministic subset size; <=0 means full split")
    parser.add_argument("--selection_seed", type=int, default=20260511)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--prompt_style", choices=["completion", "instruction"], default="completion")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--suppress_eos", action="store_true", help="Forbid EOS and PAD during generation; useful as a forced-generation sensitivity pass.")
    parser.add_argument("--suppress_special_tokens", action="store_true", help="Forbid valid tokenizer special tokens during generation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--out", required=True, help="Per-example JSONL output path")
    parser.add_argument("--summary_out", default=None, help="Aggregate JSON output path")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_available(lang: str) -> tuple[str, dict[str, Any]]:
    configs = get_dataset_config_names("csebuetnlp/xlsum", trust_remote_code=True)
    desired = LANG_TO_XLSUM[lang]
    availability = {
        "dataset": "csebuetnlp/xlsum",
        "checked_configs": configs,
        "requested_lang": lang,
        "requested_config": desired,
        "available": desired in configs,
        "project_lang_availability": {
            tag: config for tag, config in LANG_TO_XLSUM.items() if config in configs
        },
        "project_lang_unavailable": {
            tag: config for tag, config in LANG_TO_XLSUM.items() if config not in configs
        },
    }
    if desired not in configs:
        raise SystemExit(
            f"XL-Sum config {desired!r} is unavailable for lang {lang!r}; "
            f"available project languages: {availability['project_lang_availability']}"
        )
    return desired, availability


def load_examples(args: argparse.Namespace, xlsum_config: str):
    ds = load_dataset(
        "csebuetnlp/xlsum",
        xlsum_config,
        split=args.split,
        trust_remote_code=True,
    )
    original_size = len(ds)
    if args.limit and args.limit > 0:
        selected_size = min(args.limit, original_size)
        ds = ds.shuffle(seed=args.selection_seed).select(range(selected_size))
    else:
        selected_size = original_size
    return ds, original_size, selected_size


def load_model(args: argparse.Namespace):
    dtype = torch_dtype(args.dtype)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    return model, tokenizer, getattr(config, "_commit_hash", None)


def crop_article_for_prompt(
    tokenizer,
    lang: str,
    article: str,
    max_input_tokens: int,
    prompt_style: str,
    use_chat_template: bool,
) -> str:
    if prompt_style == "instruction":
        prefix, suffix = INSTRUCTION_PREFIXES[lang], ""
    else:
        prefix, suffix = PROMPTS[lang]
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    budget = max(16, max_input_tokens - len(prefix_ids) - len(suffix_ids) - 4)
    full_article_ids = tokenizer(article, add_special_tokens=False).input_ids
    while True:
        article_ids = full_article_ids[:budget]
        cropped_article = tokenizer.decode(article_ids, skip_special_tokens=True)
        user_text = f"{prefix}{cropped_article}{suffix}"
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = user_text
        prompt_tokens = tokenizer(prompt, add_special_tokens=True).input_ids
        if len(prompt_tokens) <= max_input_tokens or budget <= 16:
            return prompt
        budget = max(16, int(budget * 0.9))


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_predictions(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    predictions: list[str] = []
    for batch in tqdm(list(batched(rows, args.batch_size)), desc=f"Generating {args.model_key}/{args.lang}"):
        prompts = [row["prompt"] for row in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(args.device)
        prompt_width = enc.input_ids.shape[1]
        with torch.inference_mode():
            suppress_ids: set[int] = set()
            if args.suppress_eos:
                suppress_ids.update(token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None)
            if args.suppress_special_tokens:
                vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
                suppress_ids.update(
                    token_id for token_id in tokenizer.all_special_ids if 0 <= token_id < vocab_size
                )
            eos_token_id = None if tokenizer.eos_token_id in suppress_ids else tokenizer.eos_token_id
            bad_words_ids = None
            if suppress_ids:
                bad_words_ids = [[token_id] for token_id in sorted(suppress_ids)]
            output_ids = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
                bad_words_ids=bad_words_ids,
                use_cache=True,
            )
        new_ids = output_ids[:, prompt_width:]
        predictions.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return [pred.strip() for pred in predictions]


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: list[str], n: int) -> float:
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(tokens: list[str], n: int) -> float:
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def copy_ngram_precision(pred_tokens: list[str], source_tokens: list[str], n: int) -> float:
    pred_grams = ngrams(pred_tokens, n)
    if not pred_grams:
        return 0.0
    source_grams = set(ngrams(source_tokens, n))
    if not source_grams:
        return 0.0
    return sum(1 for gram in pred_grams if gram in source_grams) / len(pred_grams)


def score_prediction(
    pred: str,
    reference: str,
    source: str,
    leading_label_echo: bool,
    scorer: rouge_scorer.RougeScorer,
    chrf: CHRF,
) -> dict[str, float]:
    pred_words = words(pred)
    ref_words = words(reference)
    source_words = words(source)
    rouge = scorer.score(reference, pred)
    ref_len = len(ref_words)
    pred_len = len(pred_words)
    metrics = {
        "rouge1_f": rouge["rouge1"].fmeasure,
        "rouge2_f": rouge["rouge2"].fmeasure,
        "rougeL_f": rouge["rougeL"].fmeasure,
        "chrf": chrf.sentence_score(pred, [reference]).score,
        "pred_words": float(pred_len),
        "ref_words": float(ref_len),
        "length_ratio": pred_len / ref_len if ref_len else 0.0,
        "distinct_1": distinct_n(pred_words, 1),
        "distinct_2": distinct_n(pred_words, 2),
        "repeat_3gram_rate": repeated_ngram_rate(pred_words, 3),
        "source_copy_4gram_precision": copy_ngram_precision(pred_words, source_words, 4),
        "empty": float(len(pred.strip()) == 0),
        "leading_label_echo": float(leading_label_echo),
    }
    return metrics


def clean_prediction(lang: str, pred: str) -> tuple[str, bool]:
    cleaned = pred.strip()
    labels = SUMMARY_LABELS[lang]
    for label in labels:
        if cleaned.startswith(label):
            return cleaned[len(label) :].lstrip(), True
    return cleaned, False


def aggregate(records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    metrics = {key: [record["metrics"][key] for record in records] for key in metric_keys}
    means = {key: float(np.mean(values)) for key, values in metrics.items()}
    stderrs = {
        f"{key}_stderr": float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
        for key, values in metrics.items()
    }
    return {**means, **stderrs}


def main() -> None:
    args = parse_args()
    if args.lang not in PROMPTS:
        desired = LANG_TO_XLSUM[args.lang]
        raise SystemExit(f"No prompt template for {args.lang}/{desired}; add one before evaluating.")

    xlsum_config, availability = ensure_available(args.lang)
    dataset, original_size, selected_size = load_examples(args, xlsum_config)
    model, tokenizer, model_commit = load_model(args)

    rows: list[dict[str, Any]] = []
    for row in dataset:
        source = row["text"]
        prompt = crop_article_for_prompt(
            tokenizer,
            args.lang,
            source,
            args.max_input_tokens,
            args.prompt_style,
            args.use_chat_template,
        )
        rows.append(
            {
                "id": row["id"],
                "url": row["url"],
                "title": row["title"],
                "source_text": source,
                "reference_summary": row["summary"],
                "prompt": prompt,
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=True).input_ids),
            }
        )

    raw_predictions = generate_predictions(model, tokenizer, rows, args)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    chrf = CHRF(word_order=2)
    records: list[dict[str, Any]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for row, raw_pred in zip(rows, raw_predictions):
            pred, leading_label_echo = clean_prediction(args.lang, raw_pred)
            metrics = score_prediction(
                pred,
                row["reference_summary"],
                row["source_text"],
                leading_label_echo,
                scorer,
                chrf,
            )
            record = {
                "model_id": args.model_id,
                "model_key": args.model_key,
                "model_label": args.model_label,
                "model_commit": model_commit,
                "dataset": "csebuetnlp/xlsum",
                "xlsum_config": xlsum_config,
                "lang": args.lang,
                "split": args.split,
                "selection_seed": args.selection_seed,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": args.min_new_tokens,
                "prompt_style": args.prompt_style,
                "use_chat_template": args.use_chat_template,
                "decode": "greedy",
                "suppress_eos": args.suppress_eos,
                "suppress_special_tokens": args.suppress_special_tokens,
                "dtype": args.dtype,
                **row,
                "raw_prediction": raw_pred,
                "prediction": pred,
                "metrics": metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    metric_keys = [
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
        "chrf",
        "pred_words",
        "ref_words",
        "length_ratio",
        "distinct_1",
        "distinct_2",
        "repeat_3gram_rate",
        "source_copy_4gram_precision",
        "empty",
        "leading_label_echo",
    ]
    summary = {
        "model_id": args.model_id,
        "model_key": args.model_key,
        "model_label": args.model_label,
        "model_commit": model_commit,
        "dataset": "csebuetnlp/xlsum",
        "xlsum_config": xlsum_config,
        "lang": args.lang,
        "split": args.split,
        "original_split_size": original_size,
        "selected_size": selected_size,
        "limit": args.limit,
        "selection_seed": args.selection_seed,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "prompt_style": args.prompt_style,
        "use_chat_template": args.use_chat_template,
        "decode": "greedy",
        "suppress_eos": args.suppress_eos,
        "suppress_special_tokens": args.suppress_special_tokens,
        "dtype": args.dtype,
        "availability": availability,
        "example_ids": [record["id"] for record in records],
        "metrics": aggregate(records, metric_keys),
        "jsonl_path": str(out_path),
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(
        f"{args.model_key}/{args.lang}: "
        f"n={selected_size} rougeL={summary['metrics']['rougeL_f']:.4f} "
        f"chrF={summary['metrics']['chrf']:.2f} "
        f"repeat3={summary['metrics']['repeat_3gram_rate']:.4f} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
