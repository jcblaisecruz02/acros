#!/usr/bin/env python
"""Score WSD by matching base-LM hidden states between contexts and glosses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from score_wsd_gloss_activation import (  # noqa: E402
    ActivationRequest,
    activation_key,
    batched,
    build_context_request,
    build_gloss_request,
    prepare_eval_specs,
    score_predictions,
    token_positions_for_request,
    truncate,
)
from score_wsd_gloss_lm import (  # noqa: E402
    load_sense_glosses,
    load_wordnet_index,
    load_wordnet_inventory,
)
from score_wsd_mfs import (  # noqa: E402
    WSDInstance,
    lemma_pos_key,
    load_instances,
    load_wordnet_first_sense,
)


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raganato-root", type=Path, default=Path("data/wsd/WSD_Evaluation_Framework"))
    p.add_argument("--eval", action="append", default=None, help="Repeatable label=xml:gold spec.")
    p.add_argument("--wordnet-dict-dir", type=Path, default=Path("data/wsd/dict"))
    p.add_argument("--wordnet-index-sense", type=Path, default=Path("data/wsd/dict/index.sense"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", type=Path, default=Path("eval_logs/wsd/hidden_gloss_base_smollm2_360m.json"))
    p.add_argument("--activation-cache", type=Path, default=None)
    p.add_argument("--overwrite-activation-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--max-eval-instances", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--layers",
        default="8,16,24,32",
        help=(
            "Comma-separated hidden-state indices to evaluate. Use positive indices from "
            "Transformers hidden_states where 0 is embeddings and num_layers is final. "
            "Also supports 'final', 'embedding', and 'all'."
        ),
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


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


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


def parse_layers(layer_spec: str, num_hidden_layers: int) -> List[Tuple[str, int]]:
    num_hidden_states = num_hidden_layers + 1
    if layer_spec.strip().lower() == "all":
        return [(f"layer_{idx}", idx) for idx in range(num_hidden_states)]

    parsed: List[Tuple[str, int]] = []
    seen: set[int] = set()
    for raw_part in layer_spec.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "final":
            idx = num_hidden_states - 1
        elif part == "embedding":
            idx = 0
        else:
            idx = int(part)
            if idx < 0:
                idx = num_hidden_states + idx
        if idx < 0 or idx >= num_hidden_states:
            raise ValueError(
                f"Layer {raw_part!r} normalized to {idx}, outside hidden_states length {num_hidden_states}."
            )
        if idx in seen:
            continue
        seen.add(idx)
        parsed.append((f"layer_{idx}", idx))
    if not parsed:
        raise ValueError("--layers produced no layers to evaluate.")
    return parsed


def load_or_extract_hidden_vectors(
    model,
    tokenizer,
    requests: Sequence[ActivationRequest],
    layers: Sequence[Tuple[str, int]],
    args: argparse.Namespace,
    desc: str,
    cache_kind: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Counter, Dict]:
    if args.activation_cache and args.activation_cache.exists() and not args.overwrite_activation_cache:
        payload = torch.load(args.activation_cache, map_location="cpu")
        for field, requested in (
            ("model", args.model),
            ("probe_format", args.probe_format),
            ("gloss_source", args.gloss_source),
            ("normalize_activations", args.normalize_activations),
            ("max_length", args.max_length),
        ):
            cached = payload.get(field)
            if cached != requested:
                raise ValueError(
                    f"Activation cache {args.activation_cache} has {field}={cached!r}, "
                    f"but this run requested {requested!r}. Use a matching cache or "
                    "--overwrite-activation-cache."
                )
        cached_layers = payload.get("layers")
        requested_layers = [label for label, _ in layers]
        if cached_layers != requested_layers:
            raise ValueError(
                f"Activation cache {args.activation_cache} has layers={cached_layers!r}, "
                f"but this run requested {requested_layers!r}."
            )
        cached_kind = payload.get("cache_kind")
        if cached_kind != cache_kind:
            raise ValueError(
                f"Activation cache {args.activation_cache} has cache_kind={cached_kind!r}, "
                f"but this run requested {cache_kind!r}."
            )
        return payload["activations_by_layer"], Counter(), {"loaded_from_cache": str(args.activation_cache)}

    activations_by_layer, stats = extract_hidden_vectors(model, tokenizer, requests, layers, args, desc)
    cache_info = {"loaded_from_cache": None}
    if args.activation_cache:
        args.activation_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_kind": cache_kind,
                "model": args.model,
                "layers": [label for label, _ in layers],
                "probe_format": args.probe_format,
                "gloss_source": args.gloss_source,
                "normalize_activations": args.normalize_activations,
                "max_length": args.max_length,
                "activations_by_layer": activations_by_layer,
            },
            args.activation_cache,
        )
        cache_info["saved_to_cache"] = str(args.activation_cache)
    return activations_by_layer, stats, cache_info


@torch.no_grad()
def extract_hidden_vectors(
    model,
    tokenizer,
    requests: Sequence[ActivationRequest],
    layers: Sequence[Tuple[str, int]],
    args: argparse.Namespace,
    desc: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Counter]:
    device = torch.device(args.device)
    activations_by_layer: Dict[str, Dict[str, torch.Tensor]] = {label: {} for label, _ in layers}
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

        out = model(
            **enc,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Model output does not expose hidden_states.")

        row_positions: List[List[int]] = []
        row_seq_lens: List[int] = []
        for row_idx, req in enumerate(batch):
            seq_len = int(enc["attention_mask"][row_idx].sum().item())
            positions = token_positions_for_request(req, offsets[row_idx], seq_len)
            row_seq_lens.append(seq_len)
            row_positions.append(positions)
            if not positions:
                stats["missing_token_alignment"] += 1

        for label, idx in layers:
            layer_h = hidden_states[idx].detach().float()
            for row_idx, req in enumerate(batch):
                positions = row_positions[row_idx]
                if not positions:
                    continue
                valid_positions = [pos for pos in positions if pos < row_seq_lens[row_idx]]
                if not valid_positions:
                    stats["positions_truncated"] += 1
                    continue
                vec = layer_h[row_idx, valid_positions].mean(dim=0).cpu()
                if args.normalize_activations:
                    vec = vec / vec.norm(p=2).clamp_min(1e-12)
                activations_by_layer[label][req.key] = vec
            stats[f"extracted_{label}"] += sum(1 for positions in row_positions if positions)

    return activations_by_layer, stats


def build_gloss_requests(
    eval_instances: Sequence[WSDInstance],
    inventory: Mapping[str, Sequence[str]],
    sense_glosses: Mapping[str, str],
    probe_format: str,
) -> List[ActivationRequest]:
    needed_sense_keys = sorted(
        {
            sense_key
            for inst in eval_instances
            for sense_key in inventory.get(lemma_pos_key(inst.lemma, inst.pos), ())
            if sense_key in sense_glosses
        }
    )
    return [
        build_gloss_request(
            sense_key,
            lemma=sense_key.split("%", 1)[0],
            gloss=sense_glosses[sense_key],
            probe_format=probe_format,
        )
        for sense_key in needed_sense_keys
    ]


def main() -> None:
    args = parse_args()
    eval_specs = prepare_eval_specs(args)
    eval_instances = truncate(load_instances(eval_specs, kind="eval"), args.max_eval_instances)

    index_rows = load_wordnet_index(args.wordnet_index_sense)
    inventory = load_wordnet_inventory(index_rows)
    sense_glosses = load_sense_glosses(index_rows, args.wordnet_dict_dir, args.gloss_source)
    first_sense = load_wordnet_first_sense(args.wordnet_index_sense)

    context_requests = [req for inst in eval_instances if (req := build_context_request(inst)) is not None]
    gloss_requests = build_gloss_requests(eval_instances, inventory, sense_glosses, args.probe_format)

    model, tokenizer = load_model_and_tokenizer(args)
    layers = parse_layers(args.layers, int(model.config.num_hidden_layers))

    context_vectors_by_layer, context_stats = extract_hidden_vectors(
        model,
        tokenizer,
        context_requests,
        layers,
        args,
        desc="Extracting context hidden states",
    )
    gloss_vectors_by_layer, gloss_stats, cache_info = load_or_extract_hidden_vectors(
        model,
        tokenizer,
        gloss_requests,
        layers,
        args,
        desc="Extracting gloss hidden states",
        cache_kind="gloss",
    )

    scores_by_layer = {}
    for label, _ in layers:
        scores_by_layer[label] = score_predictions(
            eval_instances,
            context_vectors_by_layer[label],
            inventory,
            sense_glosses,
            gloss_vectors_by_layer[label],
            first_sense,
            include_predictions=args.include_predictions,
        )

    best_layer = max(
        scores_by_layer,
        key=lambda label: scores_by_layer[label]["overall"]["f1"],
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "base_hidden_state_gloss_matching",
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "eval_specs": eval_specs,
        "wordnet_dict_dir": str(args.wordnet_dict_dir),
        "wordnet_index_sense": str(args.wordnet_index_sense),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "layers": [{"label": label, "hidden_state_index": idx} for label, idx in layers],
        "probe_format": args.probe_format,
        "gloss_source": args.gloss_source,
        "normalize_activations": args.normalize_activations,
        "num_eval_instances_loaded": len(eval_instances),
        "num_context_requests": len(context_requests),
        "num_gloss_requests": len(gloss_requests),
        "context_activation_stats": dict(context_stats),
        "gloss_activation_stats": dict(gloss_stats),
        "cache": cache_info,
        "scores_by_layer": scores_by_layer,
        "best_layer": {
            "label": best_layer,
            "f1": scores_by_layer[best_layer]["overall"]["f1"],
            "accuracy": scores_by_layer[best_layer]["overall"]["accuracy"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Hidden-state gloss WSD:")
    for label, _ in layers:
        overall = scores_by_layer[label]["overall"]
        print(
            f"  {label}: F1={overall['f1']:.4f} "
            f"accuracy={overall['accuracy']:.4f} "
            f"answered={overall['answered']}/{overall['total']} "
            f"gloss_predictions={overall['gloss_predictions']} "
            f"first_sense_fallback={overall['first_sense_fallback']}"
        )
    print(f"Best layer: {best_layer} F1={scores_by_layer[best_layer]['overall']['f1']:.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
