"""Train script for selectable Transformer-like baselines on DeepSEA task."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import load_deepsea_train_valid, one_hot_to_dna_batch
from model_transformer import build_model, tokenize_dna_sequences


def log_step(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def mean_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if np.unique(col).size < 2:
            continue
        aucs.append(roc_auc_score(col, y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def infer_hf_model_name(model_id: str, hf_model_name: str | None) -> str:
    if hf_model_name:
        return hf_model_name
    if model_id == "dnabert2":
        return "zhihan1996/DNABERT-2-117M"
    if model_id == "nucleotide_transformer":
        return "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    raise ValueError(f"No default hf model for model_id={model_id}")


@torch.no_grad()
def evaluate(model, loader, device, model_id: str, tokenizer=None, max_length: int = 1024):
    model.eval()
    logits_all, y_all = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for x, y in loader:
        y = y.to(device)
        if model_id == "local_transformer":
            x = x.to(device)
            logits = model(x)
        else:
            seqs = one_hot_to_dna_batch(x)
            tok = tokenize_dna_sequences(seqs, tokenizer=tokenizer, max_length=max_length, device=device)
            logits = model(tok["input_ids"], tok["attention_mask"])
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        logits_all.append(torch.sigmoid(logits).cpu().numpy())
        y_all.append(y.cpu().numpy())

    y_prob = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    return {
        "loss": float(np.mean(losses)),
        "auc_macro": mean_auc(y_true, y_prob),
    }


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--valid-mat", required=True)
    parser.add_argument("--outdir", default="./outputs/torch_transformer")
    parser.add_argument(
        "--model-id",
        default="local_transformer",
        choices=["local_transformer", "dnabert2", "nucleotide_transformer"],
    )
    parser.add_argument("--hf-model-name", default=None, help="Optional override for HF model repo id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_step("train.py started")
    log_step(f"args={vars(args)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_step("loading datasets from MAT files")
    io_start = time.perf_counter()
    try:
        train_ds, valid_ds = load_deepsea_train_valid(args.train_mat, args.valid_mat)
    except Exception:
        log_step("ERROR while loading datasets")
        traceback.print_exc()
        raise
    log_step(
        "dataset loaded: "
        f"train_x={tuple(train_ds.x.shape)} train_y={tuple(train_ds.y.shape)} "
        f"valid_x={tuple(valid_ds.x.shape)} valid_y={tuple(valid_ds.y.shape)} "
        f"(elapsed={time.perf_counter()-io_start:.2f}s)"
    )

    log_step("building DataLoader objects")
    try:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    except Exception:
        log_step("ERROR while creating DataLoaders")
        traceback.print_exc()
        raise
    log_step("DataLoader ready")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"device={device}")

    model_kwargs = {}
    tokenizer = None
    if args.model_id == "local_transformer":
        model_kwargs.update(
            {
                "seq_len": train_ds.x.shape[1],
                "in_channels": train_ds.x.shape[2],
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
            }
        )
    else:
        hf_model_name = infer_hf_model_name(args.model_id, args.hf_model_name)
        model_kwargs["hf_model_name"] = hf_model_name
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for model_id=dnabert2/nucleotide_transformer. "
                "Install with: pip install transformers"
            ) from e
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    log_step("building model")
    model = build_model(
        model_id=args.model_id,
        num_labels=train_ds.y.shape[1],
        **model_kwargs,
    ).to(device)
    log_step("model ready")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        log_step(f"epoch {epoch}/{args.epochs} started")
        model.train()
        losses = []
        for step, (x, y) in enumerate(train_loader, start=1):
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.model_id == "local_transformer":
                x = x.to(device)
                logits = model(x)
            else:
                seqs = one_hot_to_dna_batch(x)
                tok = tokenize_dna_sequences(seqs, tokenizer=tokenizer, max_length=args.max_length, device=device)
                logits = model(tok["input_ids"], tok["attention_mask"])

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0:
                log_step(f"epoch {epoch} step {step}: train_loss={loss.item():.6f}")

        train_loss = float(np.mean(losses))
        val_metrics = evaluate(
            model,
            valid_loader,
            device,
            model_id=args.model_id,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        row = {
            "epoch": epoch,
            "model_id": args.model_id,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc_macro": val_metrics["auc_macro"],
        }
        history.append(row)
        log_step(f"epoch {epoch} finished in {time.perf_counter()-epoch_start:.2f}s")
        print(row, flush=True)

        if np.isfinite(val_metrics["auc_macro"]) and val_metrics["auc_macro"] > best_auc:
            best_auc = val_metrics["auc_macro"]
            torch.save(model.state_dict(), outdir / "best_model.pt")

    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log_step(f"training completed in {time.perf_counter()-total_start:.2f}s")


if __name__ == "__main__":
    main()
