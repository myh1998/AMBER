"""Evaluate selectable Transformer-like baselines on DeepSEA validation split."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import load_deepsea_train_valid, one_hot_to_dna_batch
from model_transformer import build_model, tokenize_dna_sequences


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--valid-mat", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--model-id",
        default="local_transformer",
        choices=["local_transformer", "dnabert2", "nucleotide_transformer"],
    )
    parser.add_argument("--hf-model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
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

    model_kwargs = {}
    tokenizer = None
    if args.model_id == "local_transformer":
        model_kwargs.update(
            {
                "seq_len": train_ds.x.shape[1],
                "in_channels": train_ds.x.shape[2],
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

    model = build_model(
        model_id=args.model_id,
        num_labels=train_ds.y.shape[1],
        **model_kwargs,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in valid_loader:
            if args.model_id == "local_transformer":
                x = x.to(device)
                logits = model(x)
            else:
                seqs = one_hot_to_dna_batch(x)
                tok = tokenize_dna_sequences(seqs, tokenizer=tokenizer, max_length=args.max_length, device=device)
                logits = model(tok["input_ids"], tok["attention_mask"])
            y_prob.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    metrics = {"model_id": args.model_id, "valid_auc_macro": mean_auc(y_true, y_prob)}
    print(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
