import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast

from tqdm.auto import tqdm

import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as ddp_all_gather
from .helpers import is_main_process, get_world_size, barrier

import math
import wandb

from .helpers import (
    log_param_breakdown,
    log_trainable_params,
    maybe_save,
)

from dataclasses import dataclass

def _dist_enabled():
    """Return True when torch.distributed is both available and initialized."""
    return dist.is_available() and dist.is_initialized()

def gather_autograd(x: torch.Tensor) -> torch.Tensor:
    """
    Autograd-friendly all_gather (PyTorch >=2.1).
    If DDP is disabled, returns x.
    """
    if not _dist_enabled():
        return x
    if ddp_all_gather is None:
        # Fallback: no autograd through remote ranks
        xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(xs, x.contiguous())
        return torch.cat(xs, dim=0)
    xs = ddp_all_gather(x)  # list of tensors; grads flow for local chunk
    return torch.cat(xs, dim=0)

def call_infonce(loss_obj, q_local: torch.Tensor, k_local: torch.Tensor):
    """
    DDP-friendly call into *your* InfoNCE object.
    - Gathers q,k across GPUs in the same order so positives remain on the diagonal.
    - Then calls loss_obj(q_global, k_global, *args, **kwargs).
    - Temperatures/schedules live entirely inside loss_obj — unchanged.
    """
    q_global = gather_autograd(q_local)
    k_global = gather_autograd(k_local)

    return loss_obj(q_global, k_global)

def reduce_mean(t: torch.Tensor, detach=True):
    """Average a tensor across ranks, optionally detaching from autograd graph."""
    if get_world_size() == 1:
        return t.detach() if detach else t
    rt = t.detach().clone() if detach else t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

def set_requires_grad(module, flag: bool):
    """Enable or disable gradient computation for every parameter in a module."""
    for p in module.parameters():
        p.requires_grad = flag

def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model

def is_compiled_model(model) -> bool:
    if hasattr(model, "module"):
        model = model.module
    return hasattr(model, "_orig_mod")

def sense_module_container(model):
    """Return the module that owns SENSiA sense components."""
    model = unwrap_model(model)
    return getattr(model, "backpack", model)

def freeze_sense_modules(model, logger=None, step: int | None = None, progress: float | None = None) -> bool:
    """Freeze sense/contextualization modules for Backpack and residual SenseLlama models."""
    container = sense_module_container(model)
    froze_any = False
    for attr, label in (
        ("sense_network", "Sense network"),
        ("sense_weight_net", "Weight network"),
    ):
        module = getattr(container, attr, None)
        if module is None:
            continue
        set_requires_grad(module, False)
        if logger:
            loc = f" at step {step} (p={progress:.3f})" if step is not None and progress is not None else ""
            logger.info(f"[Freeze] {label} frozen{loc}.")
        froze_any = True
    if froze_any:
        log_trainable_params(unwrap_model(model), logger)
        log_param_breakdown(container, logger)
    elif logger:
        logger.warning("No sense-related modules found to freeze.")
    return froze_any

def get_last_token_embedding(hidden_states, attention_mask):
    """
    Gets the last token embedding from contextualized (sequence level) backpack output
    hidden_states: [B, L, D]
    attention_mask: [B, L] with 1 for real tokens, 0 for pads
    returns: [B, D] embeddings for last non-pad token per example
    """
    # clamp ensures we don't index at -1 if a row is all pads
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1  # [B]
    gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
    return hidden_states.gather(dim=1, index=gather_idx).squeeze(1)


def get_context_hidden_states(model_output):
    """Return the sequence states used for context alignment.

    Backpack-Llama exposes these as `backpack_hidden_states`. Residual
    SenseLlama exposes the final residual LM states as `hidden_states`, while
    `backpack_hidden_states` is only the sense residual (`sense_mix`).
    """
    hidden_states = getattr(model_output, "hidden_states", None)
    if isinstance(hidden_states, (tuple, list)):
        if hidden_states:
            return hidden_states[-1]
    elif torch.is_tensor(hidden_states):
        return hidden_states

    base_hidden_states = getattr(model_output, "base_hidden_states", None)
    if torch.is_tensor(base_hidden_states):
        return base_hidden_states

    backpack_hidden_states = getattr(model_output, "backpack_hidden_states", None)
    if torch.is_tensor(backpack_hidden_states):
        return backpack_hidden_states

    raise AttributeError("Model output has no usable context hidden states.")


def clone_tensor_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tensor_tree(x) for x in value)
    if isinstance(value, list):
        return [clone_tensor_tree(x) for x in value]
    return value


def materialize_source_output_for_compile(model_output):
    """Clone source-output tensors that are consumed after a later forward.

    `torch.compile(mode="reduce-overhead")` may use CUDAGraph output buffers.
    The SENSiA loop runs source forward, then target forward, then consumes the
    source output. Cloning these fields outside the compiled graph preserves
    autograd while preventing the later forward from overwriting source tensors.
    """
    for field_name in (
        "hidden_states",
        "base_hidden_states",
        "backpack_hidden_states",
        "contextualization",
        "senses",
        "sense_mix",
    ):
        if not hasattr(model_output, field_name):
            continue
        value = getattr(model_output, field_name)
        if value is not None:
            setattr(model_output, field_name, clone_tensor_tree(value))
    return model_output


def assert_finite_losses(step: int, logger=None, **losses):
    bad = []
    for name, value in losses.items():
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        finite = torch.isfinite(value.detach()).all().item()
        if not finite:
            try:
                shown = float(value.detach().float().mean().item())
            except Exception:
                shown = value
            bad.append(f"{name}={shown}")
    if bad:
        message = f"Non-finite training loss at step {step}: " + ", ".join(bad)
        if logger is not None:
            logger.error(message)
        raise FloatingPointError(message)


def get_amp_settings(model, device: torch.device, use_mixed_precision: bool, mixed_precision_dtype: str = "fp16"):
    """
    Choose autocast settings. Grad scaling is only enabled for float16.
    """
    if (not use_mixed_precision) or device.type != "cuda":
        return False, None, False

    if mixed_precision_dtype == "auto":
        mixed_precision_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if mixed_precision_dtype == "bf16":
        return True, torch.bfloat16, False

    return True, torch.float16, True

# This has the tendency to blow up memory because of how we compute contribution per sense, just be careful
def pool_senses_to_seq_weighted(E, ctxt, token_mask, sense_pool_temp=1.0, eps=1e-8, normalize=True):
    """Soft-select senses per token, then average token representations into a sentence vector."""
    # Avoid fp16 here. In fp16, saturated sense softmaxes can make the entropy
    # path hit 0 * log(0). Prefer bf16 on supported GPUs to keep the large
    # [B,K,L,D] contribution tensor memory-light, then do probability/log math
    # in fp32.
    with autocast(device_type=E.device.type, enabled=False):
        work_dtype = torch.float32
        if E.device.type == "cuda" and torch.cuda.is_bf16_supported():
            work_dtype = torch.bfloat16
        E = E.to(work_dtype)
        ctxt = ctxt.to(work_dtype)
        token_mask = token_mask.float()

        # Per-sense contextualized contributions: [B,K,L,D]
        per_sense = torch.matmul(ctxt, E)

        # Magnitude proxy for "how much sense k matters" at each position
        contrib = per_sense.float().norm(dim=-1)              # [B,K,L]
        p_sense = torch.softmax(contrib / sense_pool_temp, dim=1)

        # Optionally compute entropy for logging
        entropy = -(p_sense * p_sense.clamp_min(eps).log()).sum(dim=1) # [B,L]
        masked_entropy = (entropy * token_mask).sum() / token_mask.sum().clamp_min(eps)

        # Weighted mix over senses at each token: [B,L,D]
        T = (per_sense * p_sense.to(per_sense.dtype).unsqueeze(-1)).sum(dim=1)

        # Masked mean over tokens to sentence vector: [B,D]
        w = token_mask.to(T.dtype).unsqueeze(-1)
        z = (T * w).sum(dim=1) / w.sum(dim=1).clamp_min(eps)
        z = z.float()

        if normalize:
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(eps)
    return z, masked_entropy

def pool_senses(src_out, tgt_out, src_ids, tgt_ids, pad_idx, normalize=True, sense_pool_temp=1.0):
    """Apply sense pooling to source and target batches and report entropies for logging."""
    src_mask = (src_ids != pad_idx)
    tgt_mask = (tgt_ids != pad_idx)
    z_src, src_masked_entropy = pool_senses_to_seq_weighted(src_out.senses, src_out.contextualization, src_mask, sense_pool_temp=sense_pool_temp, normalize=normalize)
    z_tgt, tgt_masked_entropy = pool_senses_to_seq_weighted(tgt_out.senses, tgt_out.contextualization, tgt_mask, sense_pool_temp=sense_pool_temp, normalize=normalize)
    return z_src, z_tgt, src_masked_entropy, tgt_masked_entropy

@dataclass
class LossScheduleCfg:
    """Holds the interpolation settings for context, sense, and LM loss weights/temperatures."""
    align_pct: float = 0.20
    polish_pct: float = 0.15
    w_ctx_align: float = 0.45
    w_sns_align: float = 0.55
    w_lm_align:  float = 0.02
    w_ctx_mid: float = 0.40
    w_sns_mid: float = 0.40
    w_lm_mid:  float = 0.20
    w_ctx_tail: float = 0.15
    w_sns_tail: float = 0.15
    w_lm_tail:  float = 0.70
    tau_ctx_start: float = 0.07
    tau_ctx_end:   float = 0.04
    tau_sns_start: float = 0.05
    tau_sns_end:   float = 0.03

class LossScheduler:
    """Smoothly interpolate loss weights and temperatures across alignment, joint, and polish phases."""
    def __init__(self, cfg: LossScheduleCfg):
        self.cfg = cfg

    @staticmethod
    def _cosine_interp(a: float, b: float, t: float) -> float:
        # t in [0,1]
        return b + 0.5*(a - b)*(1 + math.cos(math.pi * max(0.0, min(1.0, t))))

    def get(self, step: int, total_steps: int):
        # progress in [0,1]
        p = 0.0 if total_steps <= 0 else max(0.0, min(1.0, step / total_steps))
        a = self.cfg.align_pct                 # alignment ends at p=a
        z = 1.0 - self.cfg.polish_pct          # polish starts at p>=z

        # weights + tau by phase
        if p <= a:
            # ALIGNMENT: ramp align -> mid (smooth), tau decays start -> end
            tA = 0.0 if a == 0 else p / a
            w_ctx = self._cosine_interp(self.cfg.w_ctx_align, self.cfg.w_ctx_mid, tA)
            w_sns = self._cosine_interp(self.cfg.w_sns_align, self.cfg.w_sns_mid, tA)
            w_lm  = self._cosine_interp(self.cfg.w_lm_align,  self.cfg.w_lm_mid,  tA)

            tau_ctx = self._cosine_interp(self.cfg.tau_ctx_start, self.cfg.tau_ctx_end, tA)
            tau_sns = self._cosine_interp(self.cfg.tau_sns_start, self.cfg.tau_sns_end, tA)

        elif p < z:
            # JOINT: ramp mid -> tail (smooth), tau held at "end" (sharp but stable)
            tJ = 0.0 if (z - a) == 0 else (p - a) / (z - a)
            w_ctx = self._cosine_interp(self.cfg.w_ctx_mid, self.cfg.w_ctx_tail, tJ)
            w_sns = self._cosine_interp(self.cfg.w_sns_mid, self.cfg.w_sns_tail, tJ)
            w_lm  = self._cosine_interp(self.cfg.w_lm_mid,  self.cfg.w_lm_tail,  tJ)

            tau_ctx = self.cfg.tau_ctx_end
            tau_sns = self.cfg.tau_sns_end

        else:
            # POLISH: fixed tail weights, tau held (contrastive may be ~0 anyway)
            w_ctx, w_sns, w_lm = self.cfg.w_ctx_tail, self.cfg.w_sns_tail, self.cfg.w_lm_tail
            tau_ctx, tau_sns = self.cfg.tau_ctx_end, self.cfg.tau_sns_end

        # normalize weights to sum to 1 (avoid zero-div)
        s = max(1e-8, w_ctx + w_sns + w_lm)
        return w_ctx / s, w_sns / s, w_lm / s, tau_ctx, tau_sns

@torch.no_grad()
def recall_at1_bidir(Z_src: torch.Tensor, Z_tgt: torch.Tensor, normalize: bool = True):
    """
    Z_src: [N, D] source sentence embeddings
    Z_tgt: [N, D] target sentence embeddings (pair i is (src_i, tgt_i))
    normalize: L2-normalize before cosine (recommended)

    Returns:
        {
          'src2tgt_R1': float,
          'tgt2src_R1': float,
          'R1': float  # average of both directions
        }
    """
    assert Z_src.ndim == 2 and Z_tgt.ndim == 2 and Z_src.shape == Z_tgt.shape
    N = Z_src.size(0)
    device = Z_src.device

    if normalize:
        Z_src = F.normalize(Z_src, dim=-1)
        Z_tgt = F.normalize(Z_tgt, dim=-1)

    # cosine similarity = dot product after normalization
    sim = Z_src @ Z_tgt.T  # [N, N]

    # src -> tgt
    pred_tgt = sim.argmax(dim=1)                          # [N]
    correct = torch.arange(N, device=device)
    r1_s2t = (pred_tgt == correct).float().mean().item()

    # tgt -> src
    pred_src = sim.argmax(dim=0)                          # [N]
    r1_t2s = (pred_src == correct).float().mean().item()

    return {
        "src2tgt_R1": r1_s2t,
        "tgt2src_R1": r1_t2s,
        "R1": 0.5 * (r1_s2t + r1_t2s),
    }

def run_evaluation(
    model,
    test_loader,
    tokenizer,
    device,
    *,
    add_lm_loss: bool,
    normalize_last_token_embeds: bool,
    normalize_sense_pooling: bool,
    sense_pool_temp: float,
    label_smoothing: float,
):
    """
    Mirrors your eval semantics:
      - Two separate forwards (src/tgt) with output_hidden_states=True
      - Contextual vectors via last non-pad token of backpack_hidden_states (+ optional L2 norm)
      - Sense vectors via pool_senses (pre-context) with entropy aggregation
      - R@1 for ctx and sense (bidirectional)
      - Target-side PPL using unsmoothed loss if available
    Returns:
      ctx (dict), sns (dict), dev_ppl (float|None), mean_src_entropy (float), mean_tgt_entropy (float)
    """
    if test_loader is None:
        return (
            {"src2tgt_R1": 0.0, "tgt2src_R1": 0.0, "R1": 0.0},
            {"src2tgt_R1": 0.0, "tgt2src_R1": 0.0, "R1": 0.0},
            None,
            0.0,
            0.0,
        )

    was_training = model.training
    model.eval()
    # Rank-0-only eval must bypass DDP.forward; otherwise DDP can issue
    # collectives while non-main ranks are waiting at the post-eval barrier.
    eval_model = unwrap_model(model)

    Zs_ctx, Zt_ctx = [], []
    Zs_sense, Zt_sense = [], []
    compile_safe_outputs = is_compiled_model(model)

    total_src_entropy_sum = 0.0
    total_tgt_entropy_sum = 0.0
    total_src_token_count = 0
    total_tgt_token_count = 0

    ce_nll_total = 0.0
    ce_tok_count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            labels_tgt = None
            if add_lm_loss:
                labels_tgt = batch["input_ids_tgt"].masked_fill(
                    batch["attention_mask_tgt"] == 0, -100
                )

            # Forward (no labels for src; masked labels for tgt if LM enabled)
            src_out = eval_model(
                input_ids=batch["input_ids_src"],
                attention_mask=batch["attention_mask_src"],
                output_hidden_states=True,
            )
            if compile_safe_outputs:
                src_out = materialize_source_output_for_compile(src_out)
            tgt_out = eval_model(
                input_ids=batch["input_ids_tgt"],
                attention_mask=batch["attention_mask_tgt"],
                labels=labels_tgt,
                output_hidden_states=True,
                label_smoothing=label_smoothing,
            )

            # Contextualized sentence vectors (last non-pad token)
            z_src = get_last_token_embedding(
                get_context_hidden_states(src_out), batch["attention_mask_src"]
            )
            z_tgt = get_last_token_embedding(
                get_context_hidden_states(tgt_out), batch["attention_mask_tgt"]
            )
            if normalize_last_token_embeds:
                z_src = F.normalize(z_src, dim=-1)
                z_tgt = F.normalize(z_tgt, dim=-1)
            Zs_ctx.append(z_src.cpu()); Zt_ctx.append(z_tgt.cpu())

            # Sense-pooled sentence vectors (pre-context) + entropies
            zs_sense, zt_sense, src_masked_entropy, tgt_masked_entropy = pool_senses(
                src_out,
                tgt_out,
                batch["input_ids_src"],
                batch["input_ids_tgt"],
                tokenizer.pad_token_id,
                normalize=normalize_sense_pooling,
                sense_pool_temp=sense_pool_temp,
            )
            Zs_sense.append(zs_sense.cpu()); Zt_sense.append(zt_sense.cpu())

            # Entropy aggregation (token-weighted)
            src_mask = (batch["input_ids_src"] != tokenizer.pad_token_id)
            tgt_mask = (batch["input_ids_tgt"] != tokenizer.pad_token_id)
            total_src_entropy_sum += float(src_masked_entropy.item() * src_mask.sum().item())
            total_tgt_entropy_sum += float(tgt_masked_entropy.item() * tgt_mask.sum().item())
            total_src_token_count += int(src_mask.sum().item())
            total_tgt_token_count += int(tgt_mask.sum().item())

            # PPL accounting (prefer unsmoothed loss; else CE sum over logits)
            if add_lm_loss and getattr(tgt_out, "loss", None) is not None:
                if hasattr(tgt_out, "loss_unsmoothed"):
                    valid_tok = int((labels_tgt[..., 1:] != -100).sum().item())
                    ce_nll_total += float(tgt_out.loss_unsmoothed) * max(1, valid_tok)
                    ce_tok_count += valid_tok
                else:
                    shift_logits = tgt_out.logits[:, :-1, :].contiguous()
                    shift_labels = labels_tgt[:, 1:].contiguous()
                    ce_sum = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    ce_nll_total += float(ce_sum.item())
                    ce_tok_count += int((shift_labels != -100).sum().item())

    # Stack & score
    Zs_ctx = torch.cat(Zs_ctx, dim=0) if len(Zs_ctx) else torch.empty(0, device="cpu")
    Zt_ctx = torch.cat(Zt_ctx, dim=0) if len(Zt_ctx) else torch.empty(0, device="cpu")
    Zs_sense = torch.cat(Zs_sense, dim=0) if len(Zs_sense) else torch.empty(0, device="cpu")
    Zt_sense = torch.cat(Zt_sense, dim=0) if len(Zt_sense) else torch.empty(0, device="cpu")

    ctx = recall_at1_bidir(Zs_ctx, Zt_ctx) if Zs_ctx.numel() and Zt_ctx.numel() else {"src2tgt_R1": 0.0, "tgt2src_R1": 0.0, "R1": 0.0}
    sns = recall_at1_bidir(Zs_sense, Zt_sense) if Zs_sense.numel() and Zt_sense.numel() else {"src2tgt_R1": 0.0, "tgt2src_R1": 0.0, "R1": 0.0}

    dev_ppl = math.exp(ce_nll_total / ce_tok_count) if (add_lm_loss and ce_tok_count > 0) else None
    mean_src_entropy = total_src_entropy_sum / max(1, total_src_token_count)
    mean_tgt_entropy = total_tgt_entropy_sum / max(1, total_tgt_token_count)

    if was_training:
        model.train()

    return ctx, sns, dev_ppl, float(mean_src_entropy), float(mean_tgt_entropy)


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    lr_scheduler,
    tokenizer,
    device,
    *,
    total_steps: int,
    context_loss_fn,
    sense_loss_fn,
    scheduler: LossScheduler,
    add_lm_loss: bool = False,
    normalize_last_token_embeds: bool = True,
    normalize_sense_pooling: bool = True,
    freeze_sense_during_polish: bool = False,
    sense_pool_temp: float = 1.0,
    clip_grad_norm: float = 1.0,
    label_smoothing: float = 0.0,
    use_fp16: bool = True,
    mixed_precision_dtype: str = "fp16",
    logger=None,
    log_every_n_steps: int = 0,
    eval_every_n_steps: int = 0,   # ← NEW
    use_wandb: bool = False,
    start_step: int = 0,
    args=None,             # ← NEW
):
    """
    Fully step-based training loop with the same forward computations as before.
    """
    amp_enabled, amp_dtype, use_grad_scaler = get_amp_settings(
        model,
        device,
        use_fp16,
        mixed_precision_dtype=mixed_precision_dtype,
    )
    if logger is not None:
        logger.info(f"[AMP] enabled={amp_enabled} dtype={amp_dtype} grad_scaler={use_grad_scaler}")
    scaler = GradScaler("cuda", enabled=use_grad_scaler)

    # Ensure pad id exists for CE masking
    base_model = unwrap_model(model)
    if base_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    global_step = start_step
    total_train_loss = 0.0
    ce_nll_total = 0.0
    ce_tok_count = 0

    progress_bar = tqdm(total=total_steps, desc="Training (steps)", leave=True, initial=start_step, disable=not is_main_process(),)

    # for Polish freeze
    in_polish = False
    polish_start = 1.0 - scheduler.cfg.polish_pct if hasattr(scheduler, "cfg") else 1.0
    compile_safe_outputs = is_compiled_model(model)

    train_iter = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                epoch = getattr(train_model, "_epoch", 0) + 1
                setattr(train_model, "_epoch", epoch)
                sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_step += 1
        # Progress in [0,1] across the whole run
        p = 0.0 if total_steps <= 0 else max(0.0, min(1.0, global_step / total_steps))

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        # Freeze sense-related modules once when we enter Polish
        if freeze_sense_during_polish and (not in_polish) and (p >= polish_start):
            in_polish = freeze_sense_modules(model, logger=logger, step=global_step, progress=p)

        # Scheduler: weights + temperatures (preserve original API)
        w_ctx, w_sns, w_lm, tau_ctx, tau_sns = scheduler.get(global_step, total_steps)
        context_loss_fn.temperature = tau_ctx
        sense_loss_fn.temperature   = tau_sns

        # Prepare labels for the CE path (target side)
        labels_tgt_default = None
        if add_lm_loss:
            labels_tgt_default = batch["input_ids_tgt"].masked_fill(batch["attention_mask_tgt"] == 0, -100)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            # === Forward: source & target (mirrors original) ===
            # contextual / sense forward (no labels for src)
            src_out = model(
                input_ids=batch["input_ids_src"],
                attention_mask=batch["attention_mask_src"],
                output_hidden_states=True
            )
            if compile_safe_outputs:
                src_out = materialize_source_output_for_compile(src_out)
            tgt_out = model(
                input_ids=batch["input_ids_tgt"],
                attention_mask=batch["attention_mask_tgt"],
                labels=labels_tgt_default,    # None if add_lm_loss=False
                output_hidden_states=True,
                label_smoothing=label_smoothing
            )

            # contextualized sentence vectors (last non-pad token)
            z_src_ctx = get_last_token_embedding(get_context_hidden_states(src_out), batch["attention_mask_src"])
            z_tgt_ctx = get_last_token_embedding(get_context_hidden_states(tgt_out), batch["attention_mask_tgt"])
            if normalize_last_token_embeds:
                z_src_ctx = F.normalize(z_src_ctx, dim=-1)
                z_tgt_ctx = F.normalize(z_tgt_ctx, dim=-1)
            #contextualized_loss = context_loss_fn(z_src_ctx, z_tgt_ctx)
            contextualized_loss = call_infonce(context_loss_fn, z_src_ctx, z_tgt_ctx)

            # sense-pooled sentence vectors (pre-context) + entropies
            z_src_sense, z_tgt_sense, src_masked_entropy, tgt_masked_entropy = pool_senses(
                src_out, tgt_out,
                batch["input_ids_src"], batch["input_ids_tgt"],
                tokenizer.pad_token_id,
                normalize=normalize_sense_pooling,
                sense_pool_temp=sense_pool_temp
            )
            #sense_loss = sense_loss_fn(z_src_sense, z_tgt_sense)
            sense_loss = call_infonce(sense_loss_fn, z_src_sense, z_tgt_sense)

            # LM loss (target)
            lm_loss = torch.tensor(0.0, device=device)
            if add_lm_loss:
                lm_loss = tgt_out.loss if getattr(tgt_out, "loss", None) is not None else torch.tensor(0.0, device=device)

            # Total objective
            total_loss = w_ctx * contextualized_loss + w_sns * sense_loss + w_lm * lm_loss

        assert_finite_losses(
            global_step,
            logger=logger,
            context_loss=contextualized_loss,
            sense_loss=sense_loss,
            lm_loss=lm_loss,
            total_loss=total_loss,
            src_masked_entropy=src_masked_entropy,
            tgt_masked_entropy=tgt_masked_entropy,
        )

        # Backward + optimizer step
        optimizer_step_ran = True
        if use_grad_scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad_norm and clip_grad_norm > 0:
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=clip_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()
            optimizer_step_ran = scale_after_step >= scale_before_step
        else:
            total_loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=clip_grad_norm)
            optimizer.step()

        if lr_scheduler is not None and optimizer_step_ran:
            lr_scheduler.step()
        elif lr_scheduler is not None and logger is not None and is_main_process():
            logger.warning(
                "[AMP] Skipped optimizer/lr_scheduler step at training step %d because GradScaler detected non-finite gradients.",
                global_step,
            )

        # Maybe save
        if is_main_process():
            extra_state = {"wandb_run_id": wandb.run.id if wandb.run else None}
            maybe_save(
                args,
                global_step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                context_loss_fn=context_loss_fn,
                sense_loss_fn=sense_loss_fn,
                loss_scheduler=scheduler,
                logger=logger,
                extra_state=extra_state,
            )

        # Accounting for logs
        c_loss = float(contextualized_loss.detach())
        s_loss = float(sense_loss.detach())
        l_loss = float(lm_loss.detach()) if add_lm_loss else 0.0
        t_loss = float(total_loss.detach())
        src_masked_entropy = float(src_masked_entropy.detach())
        tgt_masked_entropy = float(tgt_masked_entropy.detach())
        total_train_loss += t_loss

        # CE-only PPL accounting
        if add_lm_loss and getattr(tgt_out, "loss", None) is not None:
            lm_loss_report = tgt_out.loss_unsmoothed
            valid_tok = (labels_tgt_default[..., 1:] != -100).sum().item()
            ce_nll_total += float(lm_loss_report) * max(1, valid_tok)
            ce_tok_count += valid_tok

        # Nice tqdm display (same keys)
        if is_main_process():
            progress_bar.set_postfix({
                "ctx_loss": f"{c_loss:.4f}",
                "sns_loss": f"{s_loss:.4f}",
                "lm_loss": f"{l_loss:.4f}" if add_lm_loss else "–",
                "β": f"C{w_ctx:.2f}/S{w_sns:.2f}/L{w_lm:.2f}",
                "τ": f"{tau_ctx:.3f}/{tau_sns:.3f}",
                "tot_loss": f"{t_loss:.4f}",
                "SF": "1" if in_polish else "0",
                "step": f"{global_step}",
            })
            progress_bar.update(1)

        # Periodic logging (keep keys)
        if is_main_process() and use_wandb and log_every_n_steps and (global_step % log_every_n_steps == 0):
            rough_ppl = math.exp(ce_nll_total / ce_tok_count) if (add_lm_loss and ce_tok_count > 0) else 0.0
            wandb.log({
                "global_step": global_step,
                "train/lr": optimizer.param_groups[0]["lr"],
                "metrics/context_loss": c_loss,
                "metrics/sense_loss": s_loss,
                "metrics/lm_loss": l_loss if add_lm_loss else 0.0,
                "metrics/total_loss": t_loss,
                "metrics/src_masked_entropy": float(src_masked_entropy),
                "metrics/tgt_masked_entropy": float(tgt_masked_entropy),
                "sched/weight_context": w_ctx,
                "sched/weight_sense":   w_sns,
                "sched/weight_lm":      w_lm,
                "sched/tau_context":    tau_ctx,
                "sched/tau_sense":      tau_sns,
                "metrics/rough_ppl":      rough_ppl,
            }, step=global_step)

            # reset ppl window for the next log interval
            ce_nll_total, ce_tok_count = 0.0, 0

        # Evaluation
        do_eval_now = eval_every_n_steps and (global_step % eval_every_n_steps == 0)
        if do_eval_now and get_world_size() > 1:
            barrier()

        if do_eval_now:
            eval_result = None
            if test_loader is not None:
                # In DDP, every rank runs the same small eval and only rank 0 logs.
                # This keeps collective ordering identical before training resumes.
                ctx, sns, dev_ppl, mean_src_entropy, mean_tgt_entropy = run_evaluation(
                    model, test_loader, tokenizer, device,
                    add_lm_loss=add_lm_loss,
                    normalize_last_token_embeds=normalize_last_token_embeds,
                    normalize_sense_pooling=normalize_sense_pooling,
                    sense_pool_temp=sense_pool_temp,
                    label_smoothing=label_smoothing,
                )
                eval_result = (ctx, sns, dev_ppl, mean_src_entropy, mean_tgt_entropy)

            if is_main_process() and eval_result is not None:
                ctx, sns, dev_ppl, mean_src_entropy, mean_tgt_entropy = eval_result
                if logger is not None:
                    logger.info(f"[Eval] ctx R@1 s2t={ctx['src2tgt_R1']:.4f} t2s={ctx['tgt2src_R1']:.4f} avg={ctx['R1']:.4f}")
                    logger.info(f"[Eval] sns R@1 s2t={sns['src2tgt_R1']:.4f} t2s={sns['tgt2src_R1']:.4f} avg={sns['R1']:.4f}")
                    logger.info(f"[Eval] mean entropy src={mean_src_entropy:.4f} tgt={mean_tgt_entropy:.4f}")
                    if dev_ppl is not None:
                        logger.info(f"[Eval] Target PPL: {dev_ppl:.2f}")

                if use_wandb:
                    wandb.log({
                        "eval/ctx_R1_src2tgt": ctx["src2tgt_R1"],
                        "eval/ctx_R1_tgt2src": ctx["tgt2src_R1"],
                        "eval/ctx_R1_avg": ctx["R1"],
                        "eval/sns_R1_src2tgt": sns["src2tgt_R1"],
                        "eval/sns_R1_tgt2src": sns["tgt2src_R1"],
                        "eval/sns_R1_avg": sns["R1"],
                        "eval/dev_ppl": dev_ppl if dev_ppl is not None else 0.0,
                        "eval/mean_src_entropy": mean_src_entropy,
                        "eval/mean_tgt_entropy": mean_tgt_entropy,
                    }, step=global_step)
            if get_world_size() > 1: barrier()

    progress_bar.close()
    if logger is not None:
        avg_loss = total_train_loss / max(1, total_steps)
        msg = f"avg_train_loss={avg_loss:.4f}"
        train_ppl = math.exp(ce_nll_total / ce_tok_count) if (add_lm_loss and ce_tok_count > 0) else None
        if train_ppl is not None:
            msg += f" | (unsmoothed) tgt_train_ppl={train_ppl:.2f}"
        logger.info(msg)

    return model
