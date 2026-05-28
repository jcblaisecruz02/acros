import argparse
import json
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.data import FLORES_CODE_MAP, load_flores_pair, preprocess_flores, collate_fn
from utils.training import (
    get_context_hidden_states,
    get_last_token_embedding,
    is_compiled_model,
    materialize_source_output_for_compile,
    pool_senses,
    run_evaluation,
    unwrap_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone post-adaptation evaluation for Backpack checkpoints.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--validation_dataset", type=str, default="facebook/flores")
    p.add_argument("--validation_split", type=str, default="devtest")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--normalize_last_token_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize_sense_pooling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--add_lm_loss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sense_pool_temp", type=float, default=0.7)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--details_out", type=str, default="")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_logger():
    logger = logging.getLogger("post_adaptation_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def retrieval_with_details(z_src: torch.Tensor, z_tgt: torch.Tensor, prefix: str) -> tuple[dict, list[dict]]:
    assert z_src.ndim == 2 and z_tgt.ndim == 2 and z_src.shape == z_tgt.shape
    z_src = F.normalize(z_src.float(), dim=-1)
    z_tgt = F.normalize(z_tgt.float(), dim=-1)
    sim = z_src @ z_tgt.T
    n = sim.size(0)
    correct = torch.arange(n)

    pred_tgt = sim.argmax(dim=1)
    pred_src = sim.argmax(dim=0)

    target_s2t = sim[correct, correct]
    rank_s2t = (sim > target_s2t.unsqueeze(1)).sum(dim=1) + 1
    target_t2s = sim[correct, correct]
    rank_t2s = (sim > target_t2s.unsqueeze(0)).sum(dim=0) + 1

    s2t_correct = pred_tgt == correct
    t2s_correct = pred_src == correct
    metrics = {
        "src2tgt_R1": s2t_correct.float().mean().item(),
        "tgt2src_R1": t2s_correct.float().mean().item(),
    }
    metrics["R1"] = 0.5 * (metrics["src2tgt_R1"] + metrics["tgt2src_R1"])

    rows = []
    for idx in range(n):
        rows.append(
            {
                "example_index": idx,
                f"{prefix}_src2tgt_correct": bool(s2t_correct[idx].item()),
                f"{prefix}_src2tgt_pred_index": int(pred_tgt[idx].item()),
                f"{prefix}_src2tgt_rank": int(rank_s2t[idx].item()),
                f"{prefix}_src2tgt_gold_score": float(target_s2t[idx].item()),
                f"{prefix}_tgt2src_correct": bool(t2s_correct[idx].item()),
                f"{prefix}_tgt2src_pred_index": int(pred_src[idx].item()),
                f"{prefix}_tgt2src_rank": int(rank_t2s[idx].item()),
                f"{prefix}_tgt2src_gold_score": float(target_t2s[idx].item()),
            }
        )
    return metrics, rows


def run_evaluation_with_details(
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
    was_training = model.training
    model.eval()
    eval_model = unwrap_model(model)
    compile_safe_outputs = is_compiled_model(model)

    zs_ctx, zt_ctx = [], []
    zs_sense, zt_sense = [], []
    ppl_rows = []
    total_src_entropy_sum = 0.0
    total_tgt_entropy_sum = 0.0
    total_src_token_count = 0
    total_tgt_token_count = 0
    ce_nll_total = 0.0
    ce_tok_count = 0
    example_offset = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            labels_tgt = None
            if add_lm_loss:
                labels_tgt = batch["input_ids_tgt"].masked_fill(
                    batch["attention_mask_tgt"] == 0, -100
                )

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

            z_src = get_last_token_embedding(
                get_context_hidden_states(src_out), batch["attention_mask_src"]
            )
            z_tgt = get_last_token_embedding(
                get_context_hidden_states(tgt_out), batch["attention_mask_tgt"]
            )
            if normalize_last_token_embeds:
                z_src = F.normalize(z_src, dim=-1)
                z_tgt = F.normalize(z_tgt, dim=-1)
            zs_ctx.append(z_src.cpu())
            zt_ctx.append(z_tgt.cpu())

            z_src_sense, z_tgt_sense, src_entropy, tgt_entropy = pool_senses(
                src_out,
                tgt_out,
                batch["input_ids_src"],
                batch["input_ids_tgt"],
                tokenizer.pad_token_id,
                normalize=normalize_sense_pooling,
                sense_pool_temp=sense_pool_temp,
            )
            zs_sense.append(z_src_sense.cpu())
            zt_sense.append(z_tgt_sense.cpu())

            src_mask = batch["input_ids_src"] != tokenizer.pad_token_id
            tgt_mask = batch["input_ids_tgt"] != tokenizer.pad_token_id
            total_src_entropy_sum += float(src_entropy.item() * src_mask.sum().item())
            total_tgt_entropy_sum += float(tgt_entropy.item() * tgt_mask.sum().item())
            total_src_token_count += int(src_mask.sum().item())
            total_tgt_token_count += int(tgt_mask.sum().item())

            if add_lm_loss and labels_tgt is not None:
                shift_logits = tgt_out.logits[:, :-1, :].float().contiguous()
                shift_labels = labels_tgt[:, 1:].contiguous()
                token_nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(shift_labels.shape)
                valid = shift_labels != -100
                row_nll = (token_nll * valid).sum(dim=1)
                row_tokens = valid.sum(dim=1)
                for local_idx in range(row_nll.size(0)):
                    nll = float(row_nll[local_idx].item())
                    tokens = int(row_tokens[local_idx].item())
                    ppl_rows.append(
                        {
                            "example_index": example_offset + local_idx,
                            "target_nll": nll,
                            "target_tokens": tokens,
                            "target_ppl": math.exp(nll / tokens) if tokens > 0 else None,
                        }
                    )
                ce_nll_total += float(row_nll.sum().item())
                ce_tok_count += int(row_tokens.sum().item())

            example_offset += batch["input_ids_src"].size(0)

    zs_ctx = torch.cat(zs_ctx, dim=0)
    zt_ctx = torch.cat(zt_ctx, dim=0)
    zs_sense = torch.cat(zs_sense, dim=0)
    zt_sense = torch.cat(zt_sense, dim=0)

    ctx, ctx_rows = retrieval_with_details(zs_ctx, zt_ctx, "ctx")
    sns, sns_rows = retrieval_with_details(zs_sense, zt_sense, "sns")
    by_index = {row["example_index"]: row for row in ctx_rows}
    for row in sns_rows:
        by_index[row["example_index"]].update(row)
    for row in ppl_rows:
        by_index.setdefault(row["example_index"], {"example_index": row["example_index"]}).update(row)

    dev_ppl = math.exp(ce_nll_total / ce_tok_count) if (add_lm_loss and ce_tok_count > 0) else None
    mean_src_entropy = total_src_entropy_sum / max(1, total_src_token_count)
    mean_tgt_entropy = total_tgt_entropy_sum / max(1, total_tgt_token_count)

    if was_training:
        model.train()

    return (
        ctx,
        sns,
        dev_ppl,
        float(mean_src_entropy),
        float(mean_tgt_entropy),
        [by_index[idx] for idx in sorted(by_index)],
    )


def main():
    args = parse_args()
    logger = build_logger()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Loading tokenizer from %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading config from %s", args.model_id)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    logger.info("Loading model from %s", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, config=config, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    flores_src = FLORES_CODE_MAP[args.src]
    flores_tgt = FLORES_CODE_MAP[args.tgt]
    logger.info(
        "Loading validation data dataset=%s split=%s src=%s tgt=%s",
        args.validation_dataset,
        args.validation_split,
        flores_src,
        flores_tgt,
    )
    flores_ds = load_flores_pair(
        flores_src,
        flores_tgt,
        split=args.validation_split,
        dataset=args.validation_dataset,
    )
    if args.limit and args.limit > 0:
        flores_ds = flores_ds.select(range(min(args.limit, len(flores_ds))))
        logger.info("Limiting validation data to %d examples", len(flores_ds))

    logger.info("Tokenizing validation set")
    tokenized_flores = flores_ds.map(
        preprocess_flores,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        },
        remove_columns=flores_ds.column_names,
    )

    logger.info("Building evaluation dataloader")
    test_dataloader = DataLoader(
        tokenized_flores,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    details = None
    logger.info("Running evaluation")
    if args.details_out:
        ctx, sns, dev_ppl, mean_src_entropy, mean_tgt_entropy, details = run_evaluation_with_details(
            model,
            test_dataloader,
            tokenizer,
            device,
            add_lm_loss=args.add_lm_loss,
            normalize_last_token_embeds=args.normalize_last_token_embeds,
            normalize_sense_pooling=args.normalize_sense_pooling,
            sense_pool_temp=args.sense_pool_temp,
            label_smoothing=args.label_smoothing,
        )
    else:
        ctx, sns, dev_ppl, mean_src_entropy, mean_tgt_entropy = run_evaluation(
            model,
            test_dataloader,
            tokenizer,
            device,
            add_lm_loss=args.add_lm_loss,
            normalize_last_token_embeds=args.normalize_last_token_embeds,
            normalize_sense_pooling=args.normalize_sense_pooling,
            sense_pool_temp=args.sense_pool_temp,
            label_smoothing=args.label_smoothing,
        )

    result = {
        "model_id": args.model_id,
        "src": args.src,
        "tgt": args.tgt,
        "validation_dataset": args.validation_dataset,
        "validation_split": args.validation_split,
        "limit": args.limit,
        "device": str(device),
        "ctx": ctx,
        "sns": sns,
        "dev_ppl": dev_ppl,
        "mean_src_entropy": mean_src_entropy,
        "mean_tgt_entropy": mean_tgt_entropy,
    }

    logger.info("[Eval] ctx R@1 s2t=%.4f t2s=%.4f avg=%.4f", ctx["src2tgt_R1"], ctx["tgt2src_R1"], ctx["R1"])
    logger.info("[Eval] sns R@1 s2t=%.4f t2s=%.4f avg=%.4f", sns["src2tgt_R1"], sns["tgt2src_R1"], sns["R1"])
    logger.info("[Eval] mean entropy src=%.4f tgt=%.4f", mean_src_entropy, mean_tgt_entropy)
    if dev_ppl is not None:
        logger.info("[Eval] Target PPL: %.2f", dev_ppl)
    else:
        logger.info("[Eval] Target PPL: None")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info("Wrote results to %s", args.out)

    if args.details_out and details is not None:
        out_dir = os.path.dirname(args.details_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        text_lookup = {
            idx: {
                "flores_id": flores_ds[idx]["id"],
                "src_sentence": flores_ds[idx]["src_sentence"],
                "tgt_sentence": flores_ds[idx]["tgt_sentence"],
            }
            for idx in range(len(flores_ds))
        }
        for row in details:
            row.update(text_lookup.get(row["example_index"], {}))
        detail_payload = {
            "summary": result,
            "bootstrap_unit": "flores_sentence_pair",
            "records": details,
        }
        with open(args.details_out, "w", encoding="utf-8") as f:
            json.dump(detail_payload, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info("Wrote details to %s", args.details_out)


if __name__ == "__main__":
    main()
