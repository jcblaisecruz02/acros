#!/usr/bin/env python
"""Initialize a residual ACROS sense checkpoint from a pretrained decoder LM.

The initialized model is the frozen base LM plus a dormant sense pathway:

    logits = lm_head(base_hidden + gate * sense_mix)

With --gate_init 0.0 the logits should match the base LM before training.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL_DIR = Path(__file__).parent / "model"
_SENSE_MODEL_DIRS = {
    "llama": _MODEL_DIR / "sense-llama",
    "pythia": _MODEL_DIR / "sense-pythia",
    "opt": _MODEL_DIR / "sense-opt",
}
_SENSE_CLASS_SPECS = {
    "llama": ("modeling_sense_llama", "SenseLlamaLMHeadModel"),
    "pythia": ("modeling_sense_pythia", "SensePythiaLMHeadModel"),
    "opt": ("modeling_sense_opt", "SenseOPTLMHeadModel"),
}


logger = logging.getLogger("sense_init")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite_output_dir", action="store_true")
    p.add_argument("--num_senses", type=int, default=32)
    p.add_argument("--sense_intermediate_scale", type=int, default=4)
    p.add_argument("--sense_arch", choices=["auto", "llama", "pythia", "opt"], default="auto")
    p.add_argument("--gate_init", type=float, default=0.0)
    p.add_argument("--torch_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--attn_implementation", default="")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify_prompt", default="The capital of France is")
    p.add_argument("--max_logit_diff", type=float, default=1e-5)
    p.add_argument("--skip_roundtrip_check", action="store_true")
    return p.parse_args()


def torch_dtype(name: str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_base_lm(args: argparse.Namespace):
    kwargs = {
        "dtype": torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    return model


def infer_sense_arch(base_lm, requested: str) -> str:
    if requested != "auto":
        return requested
    model_type = str(getattr(base_lm.config, "model_type", "")).lower()
    if model_type in {"llama", "mistral"} or "llama" in model_type:
        return "llama"
    if model_type in {"gpt_neox", "pythia"}:
        return "pythia"
    if model_type == "opt":
        return "opt"
    raise ValueError(
        f"Could not infer sense architecture from base model_type={model_type!r}. "
        "Pass --sense_arch llama, --sense_arch pythia, or --sense_arch opt."
    )


def load_sense_model_class(arch: str):
    sense_dir = _SENSE_MODEL_DIRS[arch]
    if not sense_dir.exists():
        raise FileNotFoundError(f"Missing sense model directory for {arch}: {sense_dir}")
    if str(sense_dir) not in sys.path:
        sys.path.insert(0, str(sense_dir))
    module_name, class_name = _SENSE_CLASS_SPECS[arch]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def copy_modeling_files(output_dir: Path, arch: str) -> None:
    sense_dir = _SENSE_MODEL_DIRS[arch]
    for py_file in sense_dir.glob("*.py"):
        shutil.copy(py_file, output_dir / py_file.name)
        logger.info("Copied remote-code file: %s", py_file.name)


def tied_input_output(model) -> bool:
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    return (
        input_emb is not None
        and output_emb is not None
        and hasattr(input_emb, "weight")
        and hasattr(output_emb, "weight")
        and input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
    )


def max_prompt_logit_diff(base_lm, sense_lm, tokenizer, prompt: str, device: torch.device) -> float:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {key: value.to(device) for key, value in enc.items()}
    base_lm.to(device)
    sense_lm.to(device)
    base_lm.eval()
    sense_lm.eval()
    with torch.no_grad():
        base_logits = base_lm(**enc, use_cache=False).logits.float()
        sense_logits = sense_lm(**enc, use_cache=False).logits.float()
    return float((base_logits - sense_logits).abs().max().item())


def collect_invariants(base_lm, sense_lm, tokenizer, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    gate = float(sense_lm.gate.detach().float().item())
    diff = max_prompt_logit_diff(base_lm, sense_lm, tokenizer, args.verify_prompt, device)
    return {
        "gate": gate,
        "gate_abs": abs(gate),
        "max_logit_diff": diff,
        "input_output_tied": float(tied_input_output(sense_lm)),
        "base_input_output_tied": float(tied_input_output(base_lm)),
    }


def assert_invariants(metrics: Dict[str, float], args: argparse.Namespace) -> None:
    if abs(metrics["gate"] - float(args.gate_init)) > 1e-8:
        raise AssertionError(f"Gate mismatch: {metrics['gate']} != {args.gate_init}")
    if abs(args.gate_init) == 0.0 and metrics["max_logit_diff"] > args.max_logit_diff:
        raise AssertionError(
            f"Expected base-equivalent logits at gate=0, got max diff {metrics['max_logit_diff']:.6g}"
        )
    if metrics["base_input_output_tied"] == 1.0 and metrics["input_output_tied"] != 1.0:
        raise AssertionError("Base model is tied but sense model input embeddings and LM head are not tied.")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite_output_dir:
        logger.info("Overwriting output_dir: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base LM: %s", args.base_model)
    base_lm = load_base_lm(args)
    sense_arch = infer_sense_arch(base_lm, args.sense_arch)
    sense_cls = load_sense_model_class(sense_arch)
    logger.info("Using sense architecture: %s (%s)", sense_arch, sense_cls.__name__)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Building %s (num_senses=%d, scale=%d, gate_init=%s)",
        sense_cls.__name__,
        args.num_senses,
        args.sense_intermediate_scale,
        args.gate_init,
    )
    sense_lm = sense_cls.from_base_lm(
        base_lm,
        num_senses=args.num_senses,
        sense_intermediate_scale=args.sense_intermediate_scale,
        gate_init=args.gate_init,
        freeze_backbone=True,
        freeze_lm_head=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = collect_invariants(base_lm, sense_lm, tokenizer, args, device)
    logger.info("Init invariants before save: %s", metrics)
    assert_invariants(metrics, args)

    logger.info("Saving sense model: %s", output_dir)
    sense_lm.save_pretrained(str(output_dir), safe_serialization=False)
    copy_modeling_files(output_dir, sense_arch)
    tokenizer.save_pretrained(str(output_dir))

    if not args.skip_roundtrip_check:
        logger.info("Running save/load roundtrip check")
        roundtrip_kwargs = {
            "dtype": torch_dtype(args.torch_dtype),
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            roundtrip_kwargs["attn_implementation"] = args.attn_implementation
        roundtrip = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            **roundtrip_kwargs,
        )
        roundtrip_metrics = collect_invariants(base_lm, roundtrip, tokenizer, args, device)
        logger.info("Init invariants after load: %s", roundtrip_metrics)
        assert_invariants(roundtrip_metrics, args)

    logger.info("Done. Sense checkpoint saved to: %s", output_dir)


if __name__ == "__main__":
    main()
