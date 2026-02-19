"""Evaluate a trained Transformer baseline on DeepSEA validation split."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--valid-mat", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train_ds, valid_ds = load_deepsea_train_valid(args.train_mat, args.valid_mat)
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
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            logits = model(x)
            y_prob.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    metrics = {"valid_auc_macro": mean_auc(y_true, y_prob)}
    print(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
