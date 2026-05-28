#!/usr/bin/env python
"""Residual sense-induction trainer.

This is the ASTIA residual variant:

    loss = alpha * CLM + (1 - alpha) * T^2 * KD + lambda_div * sense_div
    logits = lm_head(base_hidden + effective_gate * sense_mix)

The default run is clean: gate_init=0 and min_train_gate=0. Bootstrap controls
are present but dormant unless explicitly enabled.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import default_data_collator, get_cosine_schedule_with_warmup

import run_distillation as distill
from evals.lama_smollm2_common import DEFAULT_TOKENIZER


PROBE_WORDS = ["bank", "bat", "spring", "light", "charge", "cell", "court", "field"]
_MODEL_DIR = Path(__file__).parent / "model"
_SENSE_MODEL_DIRS = {
    "sense_llama": _MODEL_DIR / "sense-llama",
    "sense_pythia": _MODEL_DIR / "sense-pythia",
    "sense_opt": _MODEL_DIR / "sense-opt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--student_model_name_or_path", required=True)
    p.add_argument("--teacher_model_name_or_path", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite_output_dir", action="store_true")
    p.add_argument("--resume_from_checkpoint", default="")

    p.add_argument("--dataset_name", default="jcblaise/backpack-fineweb")
    p.add_argument("--dataset_config_name", default="eng_Latn")
    p.add_argument("--train_split", default="train")
    p.add_argument("--validation_split", default="validation")
    p.add_argument("--text_column_name", default="text")
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--preprocessing_num_workers", type=int, default=8)
    p.add_argument("--overwrite_cache", action="store_true")

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--gate_learning_rate", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--stop_after_steps", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--eval_kl_temperature", type=float, default=1.0)
    p.add_argument("--lambda_div", type=float, default=0.005)

    p.add_argument("--min_train_gate", type=float, default=0.0)
    p.add_argument("--train_gate_only_steps", type=int, default=0)
    p.add_argument("--aux_recon_warmup_steps", type=int, default=0)
    p.add_argument("--aux_recon_weight", type=float, default=1.0)
    p.add_argument("--stuck_gate_check_steps", type=int, default=0)
    p.add_argument("--stuck_gate_abs_threshold", type=float, default=1e-4)
    p.add_argument("--stuck_contribution_threshold", type=float, default=1e-4)
    p.add_argument("--stuck_gate_action", choices=["warn", "stop"], default="warn")
    p.add_argument("--residual_metric_block_size", type=int, default=256)

    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--attn_implementation", default="")
    p.add_argument("--student_gradient_checkpointing", action="store_true")
    p.add_argument("--freeze_base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_lm_head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_checkpointing_use_reentrant", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--ddp_timeout", type=int, default=21600)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--disable_tqdm", action="store_true")
    p.add_argument("--log_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile_memory_steps", type=int, default=0)

    p.add_argument("--lama_probe_path", default="evals/lama_smollm2.json")
    p.add_argument("--lama_max_examples", type=int, default=0)
    p.add_argument("--lama_batch_size", type=int, default=16)
    p.add_argument("--lama_max_length", type=int, default=256)
    p.add_argument("--flores_dataset", default="facebook/flores")
    p.add_argument("--flores_lang", default="eng_Latn")
    p.add_argument("--flores_split", default="dev")
    p.add_argument("--flores_block_size", type=int, default=512)
    p.add_argument("--flores_max_tokens", type=int, default=0)
    p.add_argument("--sense_probe_words", nargs="*", default=PROBE_WORDS)
    p.add_argument("--sense_template", default="The word is {}.")
    p.add_argument("--eval_on_start", action="store_true")
    p.add_argument("--sense_kill_threshold", type=float, default=0.3)
    p.add_argument("--sense_kill_patience", type=int, default=0)

    p.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"))
    p.add_argument("--run_name", default=os.environ.get("RUN_NAME", ""))
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "astia"))
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP", ""))
    p.add_argument("--wandb_tags", default=os.environ.get("WANDB_TAGS", ""))
    p.add_argument("--wandb_id", default=os.environ.get("WANDB_RUN_ID", ""))
    p.add_argument("--wandb_resume", default=os.environ.get("WANDB_RESUME", "allow"))
    p.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", ""))
    p.add_argument("--reference_baselines_path", default="evals/lama_smollm2_baselines.json")
    p.add_argument("--log_reference_metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ref_sense_offdiag_v1", type=float, default=-0.1015)
    p.add_argument("--ref_sense_offdiag_scratch", type=float, default=0.9154)

    return p.parse_args()


def sense_model_dir_for(model) -> Path:
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type not in _SENSE_MODEL_DIRS:
        raise ValueError(
            f"Unsupported sense model_type={model_type!r}; expected one of {sorted(_SENSE_MODEL_DIRS)}."
        )
    return _SENSE_MODEL_DIRS[model_type]


def copy_sense_modeling_files(output_dir: Path, model) -> None:
    sense_dir = sense_model_dir_for(model)
    for py_file in sense_dir.glob("*.py"):
        shutil.copy(py_file, output_dir / py_file.name)


def distributed_info(timeout_seconds: int) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    return use_ddp, rank, local_rank, world_size


def enable_student_gradient_checkpointing(model, args: argparse.Namespace, logger: logging.Logger) -> None:
    unwrapped = distill.unwrap_model(model)
    target = getattr(unwrapped, "model", None) or getattr(unwrapped, "gpt_neox", None)
    if target is None or not hasattr(target, "gradient_checkpointing_enable"):
        logger.warning("Student gradient checkpointing requested, but no backbone checkpointing hook was found.")
        return
    for cfg in [getattr(unwrapped, "config", None), getattr(target, "config", None)]:
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": args.gradient_checkpointing_use_reentrant}}
    try:
        target.gradient_checkpointing_enable(**kwargs)
    except TypeError:
        target.gradient_checkpointing_enable()
    logger.info("Enabled student backbone gradient checkpointing.")


def configure_sense_runtime(model, args: argparse.Namespace) -> None:
    unwrapped = distill.unwrap_model(model)
    if hasattr(unwrapped, "set_min_train_gate"):
        unwrapped.set_min_train_gate(args.min_train_gate)


def set_trainable_parameters(model, args: argparse.Namespace, logger: logging.Logger) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    unwrapped = distill.unwrap_model(model)
    for name, param in unwrapped.named_parameters():
        if name == "gate":
            param.requires_grad = True
        elif name.startswith("sense_network.") or name.startswith("sense_weight_net."):
            param.requires_grad = True
        elif name.startswith("aux_recon_proj."):
            param.requires_grad = args.aux_recon_warmup_steps > 0
        elif name.startswith("model.") or name.startswith("gpt_neox."):
            param.requires_grad = not args.freeze_base
        elif name.startswith("lm_head.") or name.startswith("embed_out."):
            param.requires_grad = not args.freeze_lm_head
        else:
            param.requires_grad = False

    gate_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    trainable: List[torch.nn.Parameter] = []
    trainable_names: List[str] = []
    frozen_params = 0
    trainable_params = 0
    for name, param in unwrapped.named_parameters():
        if param.requires_grad:
            trainable.append(param)
            trainable_names.append(name)
            trainable_params += int(param.numel())
            if name == "gate":
                gate_params.append(param)
            else:
                other_params.append(param)
        else:
            frozen_params += int(param.numel())

    if not gate_params:
        raise ValueError("Sense induction requires a trainable `gate` parameter.")
    if not trainable:
        raise ValueError("No trainable parameters remain after freeze configuration.")
    logger.info(
        "Sense trainable summary: trainable_params=%d frozen_params=%d gate_params=%d other_tensors=%d",
        trainable_params,
        frozen_params,
        sum(int(p.numel()) for p in gate_params),
        len(other_params),
    )
    logger.info("Trainable parameter names:\n%s", "\n".join(trainable_names))
    return trainable, gate_params, other_params


def build_optimizer(
    gate_params: List[torch.nn.Parameter],
    other_params: List[torch.nn.Parameter],
    args: argparse.Namespace,
):
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.learning_rate, "weight_decay": args.weight_decay})
    gate_lr = args.gate_learning_rate if args.gate_learning_rate > 0 else args.learning_rate
    groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})
    optimizer_kwargs = {}
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **optimizer_kwargs)


def tensor_float(value: Optional[torch.Tensor], default: float = float("nan")) -> float:
    if value is None:
        return default
    return float(value.detach().float().item())


def residual_metrics_from_output(out, prefix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not hasattr(out, "base_hidden_states") or out.base_hidden_states is None:
        return metrics
    base_hidden = out.base_hidden_states.detach().float()
    sense_mix = out.sense_mix.detach().float()
    effective_gate = out.effective_gate.detach().float()
    contribution = effective_gate * sense_mix
    base_norm = base_hidden.norm(dim=-1).mean()
    sense_mix_norm = sense_mix.norm(dim=-1).mean()
    contribution_norm = contribution.norm(dim=-1).mean()
    metrics[f"{prefix}/gate_value"] = tensor_float(out.gate)
    metrics[f"{prefix}/effective_gate"] = tensor_float(out.effective_gate)
    metrics[f"{prefix}/base_hidden_norm"] = float(base_norm.item())
    metrics[f"{prefix}/sense_mix_norm"] = float(sense_mix_norm.item())
    metrics[f"{prefix}/sense_contribution_norm"] = float(contribution_norm.item())
    metrics[f"{prefix}/sense_contribution_ratio"] = float((contribution_norm / base_norm.clamp_min(1e-8)).item())
    if getattr(out, "sense_weight_entropy", None) is not None:
        metrics[f"{prefix}/sense_weight_entropy"] = tensor_float(out.sense_weight_entropy)
    norms = getattr(out, "per_sense_contribution_norms", None)
    if norms is not None:
        norms = norms.detach().float()
        metrics[f"{prefix}/per_sense_contribution_norm_mean"] = float(norms.mean().item())
        metrics[f"{prefix}/per_sense_contribution_norm_std"] = float(norms.std(unbiased=False).item())
        metrics[f"{prefix}/per_sense_contribution_norm_min"] = float(norms.min().item())
        metrics[f"{prefix}/per_sense_contribution_norm_max"] = float(norms.max().item())
    return metrics


def gate_grad_norm(model) -> float:
    unwrapped = distill.unwrap_model(model)
    gate = getattr(unwrapped, "gate", None)
    if gate is None or gate.grad is None:
        return 0.0
    return float(gate.grad.detach().float().norm().item())


def zero_non_gate_grads(model) -> None:
    unwrapped = distill.unwrap_model(model)
    for name, param in unwrapped.named_parameters():
        if name != "gate" and param.grad is not None:
            param.grad.zero_()


def aux_recon_loss(student_model, student_out, global_step: int, args: argparse.Namespace) -> torch.Tensor:
    unwrapped = distill.unwrap_model(student_model)
    proj = getattr(unwrapped, "aux_recon_proj", None)
    if proj is None or args.aux_recon_warmup_steps <= 0:
        return student_out.logits.new_zeros(())
    if global_step < args.aux_recon_warmup_steps:
        projected = proj(student_out.sense_mix.to(dtype=proj.weight.dtype))
        target = student_out.base_hidden_states.detach().to(dtype=projected.dtype)
        return F.mse_loss(projected, target)
    return student_out.logits.new_zeros(()) + 0.0 * proj.weight.sum()


def total_loss(
    student_model,
    student_out,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_clm = distill.clm_loss_from_logits(student_out.logits, labels)
    loss_kd = distill.kd_loss_from_logits(student_out.logits, teacher_logits, labels, args.temperature)
    loss_div = distill.sense_diversity_loss(student_out.senses, attention_mask)
    loss_recon = aux_recon_loss(student_model, student_out, global_step, args)
    loss = (
        args.alpha * loss_clm
        + (1.0 - args.alpha) * (args.temperature**2) * loss_kd
        + args.lambda_div * loss_div
        + args.aux_recon_weight * loss_recon
    )
    metrics = {
        "train/loss_total": float(loss.detach().float().item()),
        "train/loss_clm": float(loss_clm.detach().float().item()),
        "train/loss_kd": float(loss_kd.detach().float().item()),
        "train/loss_div": float(loss_div.detach().float().item()),
        "train/loss_aux_recon": float(loss_recon.detach().float().item()),
    }
    metrics.update(residual_metrics_from_output(student_out, "train"))
    return loss, metrics


def metric_accumulate(sums: Dict[str, float], counts: Dict[str, int], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        if not math.isfinite(float(value)):
            continue
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def metric_average(sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {key: value / max(1, counts.get(key, 1)) for key, value in sums.items()}


@torch.no_grad()
def evaluate_residual_metrics(model, tokenizer, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    texts = distill.load_flores_texts(args)
    block_size = max(8, min(args.residual_metric_block_size, args.flores_block_size))
    for chunk in distill.token_chunks(tokenizer, texts, block_size, args.flores_max_tokens, device):
        with distill.autocast_context(device, args.dtype):
            out = model(input_ids=chunk, use_cache=False, output_sense_metrics=True)
        return residual_metrics_from_output(out, "eval")
    return {}


@torch.no_grad()
def run_all_evals(student, teacher, tokenizer, args: argparse.Namespace, device: torch.device, step: int) -> Dict[str, float]:
    student_unwrapped = distill.unwrap_model(student)
    metrics = distill.run_all_evals(student_unwrapped, teacher, tokenizer, args, device, step)
    metrics.update(evaluate_residual_metrics(student_unwrapped, tokenizer, args, device))
    return metrics


def train_progress_postfix(metrics: Dict[str, float]) -> Dict[str, str]:
    return {
        "loss": f"{metrics.get('train/loss_total', float('nan')):.3f}",
        "clm": f"{metrics.get('train/loss_clm', float('nan')):.3f}",
        "kd": f"{metrics.get('train/loss_kd', float('nan')):.3f}",
        "gate": f"{metrics.get('train/gate_value', float('nan')):.2e}",
        "ratio": f"{metrics.get('train/sense_contribution_ratio', float('nan')):.2e}",
        "lr": f"{metrics.get('train/lr', float('nan')):.2e}",
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def trainer_state_json(
    step: int,
    epoch: int,
    micro_step_in_epoch: int,
    optimizer,
    scheduler,
    args: argparse.Namespace,
) -> Dict:
    return {
        "step": int(step),
        "epoch": int(epoch),
        "micro_step_in_epoch": int(micro_step_in_epoch),
        "learning_rate": float(scheduler.get_last_lr()[0]),
        "warmup_steps": int(args.warmup_steps or int(args.max_steps * args.warmup_ratio)),
        "max_steps": int(args.max_steps),
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "args": vars(args),
    }


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    micro_step_in_epoch: int,
    args: argparse.Namespace,
    logger: logging.Logger,
    wandb_run_id: str = "",
) -> None:
    out = Path(args.output_dir)
    ckpt = out / f"checkpoint-{step}"
    tmp = out / f".checkpoint-{step}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    unwrapped = distill.unwrap_model(model)
    unwrapped.save_pretrained(str(tmp), safe_serialization=False)
    copy_sense_modeling_files(tmp, unwrapped)
    tokenizer.save_pretrained(str(tmp))
    json_state = trainer_state_json(step, epoch, micro_step_in_epoch, optimizer, scheduler, args)
    if wandb_run_id:
        json_state["wandb_run_id"] = wandb_run_id
    write_json(tmp / "trainer_state.json", json_state)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "micro_step_in_epoch": micro_step_in_epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": distill.capture_rng_state(),
            "args": vars(args),
            "wandb_run_id": wandb_run_id,
        },
        tmp / "trainer_state.pt",
    )
    if ckpt.exists():
        shutil.rmtree(ckpt)
    os.replace(tmp, ckpt)
    logger.info("Saved checkpoint: %s", ckpt)

    if args.save_total_limit and args.save_total_limit > 0:
        checkpoints = sorted(
            [p for p in out.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        for old in checkpoints[:-args.save_total_limit]:
            logger.info("Removing old checkpoint: %s", old)
            shutil.rmtree(old)


def maybe_resume(
    args: argparse.Namespace,
    optimizer,
    scheduler,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, int]:
    ckpt = distill.resolve_resume_checkpoint(args)
    if ckpt is None:
        return {"step": 0, "epoch": 0, "micro_step_in_epoch": 0, "wandb_run_id": ""}
    state_path = ckpt / "trainer_state.pt"
    if not state_path.exists():
        logger.warning("Checkpoint has no trainer_state.pt, only model weights will be resumed: %s", ckpt)
        return {"step": 0, "epoch": 0, "micro_step_in_epoch": 0, "wandb_run_id": ""}
    state = distill.torch_load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    distill.restore_rng_state(state.get("rng_state"), logger)
    step = int(state.get("step", 0))
    epoch = int(state.get("epoch", 0))
    micro_step_in_epoch = int(state.get("micro_step_in_epoch", 0))
    logger.info(
        "Resumed model/optimizer/scheduler/RNG from %s at step=%d epoch=%d next_micro_step=%d lr=%.3e",
        ckpt,
        step,
        epoch,
        micro_step_in_epoch,
        scheduler.get_last_lr()[0],
    )
    return {
        "step": step,
        "epoch": epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "wandb_run_id": str(state.get("wandb_run_id", "")),
    }


def stuck_gate_stop(train_metrics: Dict[str, float], args: argparse.Namespace, step: int, logger: logging.Logger) -> bool:
    if args.stuck_gate_check_steps <= 0 or step < args.stuck_gate_check_steps:
        return False
    gate = abs(float(train_metrics.get("train/gate_value", float("inf"))))
    ratio = float(train_metrics.get("train/sense_contribution_ratio", float("inf")))
    if gate < args.stuck_gate_abs_threshold and ratio < args.stuck_contribution_threshold:
        logger.warning(
            "Gate appears stuck at step %d: |gate|=%.4e < %.4e and contribution_ratio=%.4e < %.4e",
            step,
            gate,
            args.stuck_gate_abs_threshold,
            ratio,
            args.stuck_contribution_threshold,
        )
        return args.stuck_gate_action == "stop"
    logger.info(
        "Gate bootstrap check passed at step %d: |gate|=%.4e contribution_ratio=%.4e",
        step,
        gate,
        ratio,
    )
    return False


def main() -> None:
    args = parse_args()
    use_ddp, rank, local_rank, world_size = distributed_info(args.ddp_timeout)
    is_main = distill.is_main_process(rank)
    logger = distill.setup_logger(is_main)

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    distill.prepare_output_dir(args, logger, is_main)
    distill.barrier(use_ddp)

    checkpoint = distill.resolve_resume_checkpoint(args)
    student_path = str(checkpoint) if checkpoint is not None else args.student_model_name_or_path

    tokenizer = distill.load_tokenizer(args)
    if is_main:
        logger.info("Loading sense student from %s", student_path)
    student = distill.load_causal_lm(student_path, args, device)
    configure_sense_runtime(student, args)
    if hasattr(student, "tie_weights"):
        student.tie_weights()
    if is_main:
        logger.info("Student attention implementation: %s", distill.attention_implementation(student))
    if args.student_gradient_checkpointing:
        enable_student_gradient_checkpointing(student, args, logger)
    trainable_parameters, gate_params, other_params = set_trainable_parameters(student, args, logger)

    if is_main:
        logger.info("Loading teacher from %s", args.teacher_model_name_or_path)
    teacher = distill.load_causal_lm(args.teacher_model_name_or_path, args, device)
    if is_main:
        logger.info("Teacher attention implementation: %s", distill.attention_implementation(teacher))
    teacher.eval()
    teacher.requires_grad_(False)
    if is_main and args.log_memory:
        distill.log_cuda_memory(logger, device, "after model load")

    if use_ddp:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if use_ddp:
        if is_main:
            train_dataset = distill.build_lm_dataset(args, tokenizer, logger)
        distill.barrier(use_ddp)
        if not is_main:
            if args.overwrite_cache:
                args.overwrite_cache = False
            train_dataset = distill.build_lm_dataset(args, tokenizer, logger)
        distill.barrier(use_ddp)
    else:
        train_dataset = distill.build_lm_dataset(args, tokenizer, logger)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size if use_ddp else 1,
        rank=rank if use_ddp else 0,
        shuffle=True,
        seed=args.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    optimizer = build_optimizer(gate_params, other_params, args)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    resume_state = maybe_resume(args, optimizer, scheduler, device, logger)
    global_step = int(resume_state["step"])
    epoch = int(resume_state["epoch"])
    resume_micro_step_in_epoch = int(resume_state["micro_step_in_epoch"])
    if resume_state.get("wandb_run_id") and not args.wandb_id:
        args.wandb_id = str(resume_state["wandb_run_id"])

    if is_main:
        logger.info(
            "Sense induction hparams: alpha=%.3f temperature=%.3f lambda_div=%.4g lr=%.3e gate_lr=%.3e "
            "min_train_gate=%.3e train_gate_only_steps=%d aux_recon_warmup_steps=%d max_steps=%d",
            args.alpha,
            args.temperature,
            args.lambda_div,
            args.learning_rate,
            args.gate_learning_rate if args.gate_learning_rate > 0 else args.learning_rate,
            args.min_train_gate,
            args.train_gate_only_steps,
            args.aux_recon_warmup_steps,
            args.max_steps,
        )
        logger.info(
            "Freeze flags: freeze_base=%s freeze_lm_head=%s",
            args.freeze_base,
            args.freeze_lm_head,
        )

    reference_metrics = distill.build_reference_metrics(args, logger) if is_main else {}
    wandb = distill.init_wandb(args, reference_metrics, logger, global_step) if is_main else None

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    sense_violation_count = 0
    if args.eval_on_start and is_main:
        logger.info("Running initial evaluation")
        eval_metrics = run_all_evals(student, teacher, tokenizer, args, device, global_step)
        logger.info("Initial eval: %s", eval_metrics)
        distill.write_jsonl(metrics_path, eval_metrics)
        _, sense_violation_count = distill.should_stop_for_sense_collapse(
            eval_metrics, args, sense_violation_count, logger
        )
        distill.wandb_log(wandb, eval_metrics, global_step, reference_metrics)
    distill.barrier(use_ddp)

    logger.info("Starting sense induction at step %d / %d", global_step, args.max_steps)
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    running_sums: Dict[str, float] = {}
    running_counts: Dict[str, int] = {}
    running_count = 0
    stuck_checked = False

    stop_step = args.stop_after_steps if args.stop_after_steps > 0 else args.max_steps
    progress_total = min(args.max_steps, stop_step)
    progress_bar = tqdm(
        total=progress_total,
        initial=min(global_step, progress_total),
        desc="sense-induce",
        unit="step",
        dynamic_ncols=True,
        disable=(not is_main or args.disable_tqdm),
    )
    while global_step < args.max_steps and global_step < stop_step:
        sampler.set_epoch(epoch)
        if is_main and resume_micro_step_in_epoch:
            logger.info(
                "Skipping %d already-consumed microbatches in epoch %d after resume",
                resume_micro_step_in_epoch,
                epoch,
            )
        for micro_step_in_epoch, batch in enumerate(loader):
            if global_step >= args.max_steps or global_step >= stop_step:
                break
            if micro_step_in_epoch < resume_micro_step_in_epoch:
                continue

            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.get("labels", batch["input_ids"]).clone()
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(batch["input_ids"])

            profile_memory = (
                is_main
                and args.profile_memory_steps > 0
                and global_step < args.profile_memory_steps
            )
            if profile_memory:
                distill.log_cuda_memory(logger, device, f"step {global_step} before teacher")
            with torch.no_grad():
                with distill.autocast_context(device, args.dtype):
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits.detach()
            if profile_memory:
                distill.log_cuda_memory(logger, device, f"step {global_step} after teacher")

            will_step = (running_count + 1) % args.gradient_accumulation_steps == 0
            next_step = global_step + 1 if will_step else global_step
            will_log = will_step and (next_step == 1 or next_step % args.logging_steps == 0)
            with distill.autocast_context(device, args.dtype):
                student_out = student(
                    input_ids=batch["input_ids"],
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_sense_metrics=will_log,
                )
                if profile_memory:
                    distill.log_cuda_memory(logger, device, f"step {global_step} after student")
                loss, step_metrics = total_loss(student, student_out, teacher_logits, labels, attention_mask, args, global_step)
                if profile_memory:
                    distill.log_cuda_memory(logger, device, f"step {global_step} after loss")
                scaled_loss = loss / args.gradient_accumulation_steps

            sync_gradients = (running_count + 1) % args.gradient_accumulation_steps == 0
            backward_context = student.no_sync() if use_ddp and not sync_gradients else nullcontext()
            with backward_context:
                scaled_loss.backward()
            if profile_memory:
                distill.log_cuda_memory(logger, device, f"step {global_step} after backward")
            del teacher_logits, student_out, loss, scaled_loss
            metric_accumulate(running_sums, running_counts, step_metrics)
            running_count += 1

            if running_count % args.gradient_accumulation_steps == 0:
                if global_step < args.train_gate_only_steps:
                    zero_non_gate_grads(student)
                gate_grad = gate_grad_norm(student)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main:
                    progress_bar.update(1)

                train_metrics: Dict[str, float] = {}
                should_log = global_step % args.logging_steps == 0 or global_step == 1
                if should_log:
                    train_metrics = metric_average(running_sums, running_counts)
                    train_metrics["step"] = global_step
                    lrs = scheduler.get_last_lr()
                    train_metrics["train/lr"] = float(lrs[0])
                    train_metrics["train/gate_lr"] = float(lrs[-1])
                    train_metrics["train/grad_norm"] = float(grad_norm.detach().float().item())
                    train_metrics["train/gate_grad_norm"] = gate_grad
                    if args.log_memory:
                        train_metrics.update(distill.cuda_memory_metrics(device, "memory"))
                    train_metrics = distill.mean_reduce_metrics(train_metrics, device, use_ddp)
                    if is_main:
                        logger.info(
                            "step=%d loss=%.4f clm=%.4f kd=%.4f div=%.4f gate=%.4e ratio=%.4e lr=%.3e gate_lr=%.3e grad=%.3f",
                            global_step,
                            train_metrics.get("train/loss_total", float("nan")),
                            train_metrics.get("train/loss_clm", float("nan")),
                            train_metrics.get("train/loss_kd", float("nan")),
                            train_metrics.get("train/loss_div", float("nan")),
                            train_metrics.get("train/gate_value", float("nan")),
                            train_metrics.get("train/sense_contribution_ratio", float("nan")),
                            train_metrics["train/lr"],
                            train_metrics["train/gate_lr"],
                            train_metrics["train/grad_norm"],
                        )
                        distill.write_jsonl(metrics_path, train_metrics)
                        distill.wandb_log(wandb, train_metrics, global_step)
                        progress_bar.set_postfix(train_progress_postfix(train_metrics), refresh=False)
                    if args.log_memory and device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                running_sums = {}
                running_counts = {}

                stop_for_stuck_gate = False
                if (
                    should_log
                    and not stuck_checked
                    and args.stuck_gate_check_steps > 0
                    and global_step >= args.stuck_gate_check_steps
                ):
                    stop_for_stuck_gate = stuck_gate_stop(train_metrics, args, global_step, logger)
                    stuck_checked = True

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    stop_tensor = torch.zeros((), dtype=torch.int, device=device)
                    if is_main:
                        logger.info("Running evaluation at step %d", global_step)
                        eval_metrics = run_all_evals(student, teacher, tokenizer, args, device, global_step)
                        logger.info("Eval step %d: %s", global_step, eval_metrics)
                        distill.write_jsonl(metrics_path, eval_metrics)
                        stop, sense_violation_count = distill.should_stop_for_sense_collapse(
                            eval_metrics, args, sense_violation_count, logger
                        )
                        if stop:
                            stop_tensor.fill_(1)
                        distill.wandb_log(wandb, eval_metrics, global_step, reference_metrics)
                    if use_ddp:
                        dist.broadcast(stop_tensor, src=0)
                    if int(stop_tensor.item()) != 0:
                        if is_main:
                            logger.error("Stopping run because sense off-diagonal collapse threshold was exceeded.")
                        raise SystemExit(2)
                    student.train()

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if is_main:
                        save_checkpoint(
                            student,
                            tokenizer,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            micro_step_in_epoch + 1,
                            args,
                            logger,
                            wandb.run.id if wandb and wandb.run else "",
                        )
                    distill.barrier(use_ddp)

                if stop_for_stuck_gate:
                    if is_main:
                        logger.error("Stopping run because the residual gate did not bootstrap.")
                    raise SystemExit(3)
        epoch += 1
        resume_micro_step_in_epoch = 0

    if is_main:
        progress_bar.close()
        if global_step >= args.max_steps:
            logger.info("Training complete; saving final model to %s", args.output_dir)
            unwrapped = distill.unwrap_model(student)
            unwrapped.save_pretrained(args.output_dir, safe_serialization=False)
            copy_sense_modeling_files(Path(args.output_dir), unwrapped)
            tokenizer.save_pretrained(args.output_dir)
        else:
            logger.info(
                "Stopped early at step %d due to --stop_after_steps=%d; final model save skipped.",
                global_step,
                args.stop_after_steps,
            )
        if wandb:
            wandb.finish()
    distill.barrier(use_ddp)

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
