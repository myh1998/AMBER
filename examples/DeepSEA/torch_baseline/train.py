"""Train script for Transformer baseline on DeepSEA task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import load_deepsea_train_valid
from model_transformer import DeepSEATransformer


def mean_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if np.unique(col).size < 2:
            continue
        aucs.append(roc_auc_score(col, y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--valid-mat", required=True)
    parser.add_argument("--outdir", default="./outputs/torch_transformer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds, valid_ds = load_deepsea_train_valid(args.train_mat, args.valid_mat)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSEATransformer(
        seq_len=train_ds.x.shape[1],
        in_channels=train_ds.x.shape[2],
        num_labels=train_ds.y.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_metrics = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc_macro": val_metrics["auc_macro"],
        }
        history.append(row)
        print(row)

        if np.isfinite(val_metrics["auc_macro"]) and val_metrics["auc_macro"] > best_auc:
            best_auc = val_metrics["auc_macro"]
            torch.save(model.state_dict(), outdir / "best_model.pt")

    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
