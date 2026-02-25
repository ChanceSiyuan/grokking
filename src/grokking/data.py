"""Modular arithmetic dataset for grokking experiments."""

import torch
import numpy as np


def make_modular_addition_dataset(
    p: int = 97,
    train_fraction: float = 0.3,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Generate modular addition dataset: (a + b) mod p.

    Args:
        p: Prime modulus.
        train_fraction: Fraction of all p² pairs used for training.
        seed: Random seed for train/test split.
        device: Torch device.

    Returns:
        Dict with keys: train_x, train_y, test_x, test_y, p, all_a, all_b, all_y.
    """
    rng = np.random.RandomState(seed)

    # All p² input pairs
    a = np.arange(p)
    b = np.arange(p)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    all_a = aa.flatten()  # shape (p²,)
    all_b = bb.flatten()
    all_y = (all_a + all_b) % p

    # One-hot encode: concatenate one-hot(a) and one-hot(b) → dim 2p
    n = len(all_a)
    x = np.zeros((n, 2 * p), dtype=np.float32)
    x[np.arange(n), all_a] = 1.0
    x[np.arange(n), p + all_b] = 1.0

    # Random train/test split
    indices = rng.permutation(n)
    n_train = int(train_fraction * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return {
        "train_x": torch.tensor(x[train_idx], device=device),
        "train_y": torch.tensor(all_y[train_idx], dtype=torch.long, device=device),
        "test_x": torch.tensor(x[test_idx], device=device),
        "test_y": torch.tensor(all_y[test_idx], dtype=torch.long, device=device),
        "p": p,
        "all_a": all_a,
        "all_b": all_b,
        "all_y": all_y,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
