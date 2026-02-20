"""Dataset utilities for DeepSEA-style 1000bp -> 919 multi-label task."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


DNA_VOCAB = np.array(["A", "C", "G", "T"])


class DeepSEADataset(Dataset):
    """PyTorch dataset for DeepSEA matrices.

    Expected arrays:
    - X: shape (N, 1000, 4) or (N, 4, 1000)
    - y: shape (N, 919)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f"Expected x to be 3D, got shape={x.shape}")
        if y.ndim != 2:
            raise ValueError(f"Expected y to be 2D, got shape={y.shape}")

        # Fix common channel layout issue
        if x.shape[-1] != 4 and x.shape[1] == 4:
            x = np.transpose(x, (0, 2, 1))

        if x.shape[-1] != 4:
            raise ValueError(f"Expected last dim to be 4 channels, got shape={x.shape}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample counts: x={x.shape[0]} y={y.shape[0]}")

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _read_mat_file(path: Path) -> dict:
    """Read matlab file that may be HDF5-based or old MAT format."""
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            out = {k: np.array(f[k]) for k in keys}
            return out
    except OSError:
        # Non-HDF5 MAT format
        out = loadmat(path)
        return {k: v for k, v in out.items() if not k.startswith("__")}


def _pick_xy(data: dict, x_candidates, y_candidates) -> Tuple[np.ndarray, np.ndarray]:
    x_key = next((k for k in x_candidates if k in data), None)
    y_key = next((k for k in y_candidates if k in data), None)
    if x_key is None or y_key is None:
        raise KeyError(
            f"Could not locate X/Y keys in {list(data.keys())}. "
            f"X candidates={x_candidates}, Y candidates={y_candidates}"
        )
    x = data[x_key]
    y = data[y_key]

    # Normalize y to [N, 919] first.
    if y.ndim == 2 and y.shape[0] == 919 and y.shape[1] != 919:
        y = y.T

    # Normalize x to [N, 1000, 4].
    # Common DeepSEA layouts include:
    # - [N, 1000, 4]
    # - [N, 4, 1000]
    # - [1000, 4, N]
    # - [4, 1000, N]
    if x.ndim != 3:
        raise ValueError(f"Expected x to be 3D before normalization, got shape={x.shape}")

    # If sample dimension is at the end (e.g. [1000, 4, N] or [4, 1000, N]).
    if x.shape[-1] == y.shape[0]:
        if x.shape[0] == 1000 and x.shape[1] == 4:
            x = np.transpose(x, (2, 0, 1))
        elif x.shape[0] == 4 and x.shape[1] == 1000:
            x = np.transpose(x, (2, 1, 0))
        else:
            # generic: move sample axis to front and keep the other two in order
            x = np.transpose(x, (2, 0, 1))

    # If sample dimension is already first but channel/length are swapped.
    if x.shape[0] == y.shape[0] and x.shape[1] == 4 and x.shape[2] == 1000:
        x = np.transpose(x, (0, 2, 1))

    return x, y


def load_deepsea_train_valid(train_mat: str, valid_mat: str):
    train_data = _read_mat_file(Path(train_mat))
    valid_data = _read_mat_file(Path(valid_mat))

    x_train, y_train = _pick_xy(
        train_data,
        x_candidates=("trainxdata", "x", "X_train", "X"),
        y_candidates=("traindata", "y", "Y_train", "Y"),
    )
    x_valid, y_valid = _pick_xy(
        valid_data,
        x_candidates=("validxdata", "x", "X_valid", "X"),
        y_candidates=("validdata", "y", "Y_valid", "Y"),
    )

    return DeepSEADataset(x_train, y_train), DeepSEADataset(x_valid, y_valid)


def one_hot_to_dna_batch(x: torch.Tensor) -> list:
    """Convert one-hot DNA tensor [B, L, 4] to list[str] sequences."""
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError(f"Expected [B, L, 4], got shape={tuple(x.shape)}")
    idx = x.argmax(dim=-1).cpu().numpy()
    return ["".join(DNA_VOCAB[row]) for row in idx]
