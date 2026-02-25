"""Training loop for grokking experiments."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import make_modular_addition_dataset
from .model import GrokMLP
from .metrics import OrderParameterTracker


def train(
    p: int = 97,
    width: int = 512,
    alpha: float = 0.0,
    weight_decay: float = 0.1,
    lr: float = 1e-3,
    train_fraction: float = 0.3,
    n_epochs: int = 100_000,
    log_every: int = 100,
    checkpoint_epochs: list[int] | None = None,
    save_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
    compute_order_params: bool = True,
) -> dict:
    """Run a full grokking training experiment.

    Args:
        p: Prime modulus.
        width: Hidden layer width.
        alpha: muP scale parameter.
        weight_decay: AdamW weight decay lambda.
        lr: Base learning rate.
        train_fraction: Fraction of data for training.
        n_epochs: Maximum number of epochs.
        log_every: Log metrics every N epochs.
        checkpoint_epochs: Specific epochs to save model checkpoints.
        save_dir: Directory to save results (None = don't save).
        device: Torch device (None = auto-detect).
        seed: Random seed.
        compute_order_params: Whether to compute order parameters (expensive).

    Returns:
        Dict with training logs and metadata.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    # Data
    data = make_modular_addition_dataset(
        p=p, train_fraction=train_fraction, seed=seed, device=device
    )
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]
    n_train = train_x.shape[0]
    n_test = test_x.shape[0]

    # Model
    model = GrokMLP(p=p, width=width, alpha=alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracker
    if compute_order_params:
        # Use test set for order parameter computation (larger, representative)
        tracker = OrderParameterTracker(model, p, device)

    # Logging
    log = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    if compute_order_params:
        for key in ["rqi", "effective_rank", "participation_ratio",
                     "kernel_alignment", "snr"]:
            log[key] = []

    checkpoints = {}
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Detect grokking milestones
    t_mem = None  # epoch where train acc first > 99%
    t_grok = None  # epoch where test acc first > 95%

    start_time = time.time()

    pbar = tqdm(range(n_epochs), desc=f"alpha={alpha}, wd={weight_decay}")
    for epoch in pbar:
        # --- Training step (full batch) ---
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        # --- Logging ---
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_x)
                train_loss = criterion(train_logits, train_y).item()
                train_acc = (train_logits.argmax(dim=1) == train_y).float().mean().item()

                test_logits = model(test_x)
                test_loss = criterion(test_logits, test_y).item()
                test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["test_loss"].append(test_loss)
            log["train_acc"].append(train_acc)
            log["test_acc"].append(test_acc)

            if compute_order_params:
                op = tracker.compute_all(test_x, test_y)
                for key, val in op.items():
                    log[key].append(val)

            pbar.set_postfix(
                tr_acc=f"{train_acc:.3f}",
                te_acc=f"{test_acc:.3f}",
                tr_loss=f"{train_loss:.4f}",
            )

            # Milestone detection
            if t_mem is None and train_acc > 0.99:
                t_mem = epoch
            if t_grok is None and test_acc > 0.95:
                t_grok = epoch

        # --- Checkpointing ---
        if checkpoint_epochs and epoch in checkpoint_epochs:
            checkpoints[epoch] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    elapsed = time.time() - start_time

    result = {
        "log": log,
        "config": {
            "p": p,
            "width": width,
            "alpha": alpha,
            "weight_decay": weight_decay,
            "lr": lr,
            "train_fraction": train_fraction,
            "n_epochs": n_epochs,
            "seed": seed,
            "device": device,
        },
        "t_mem": t_mem,
        "t_grok": t_grok,
        "elapsed_seconds": elapsed,
        "n_train": n_train,
        "n_test": n_test,
    }

    if checkpoints:
        result["checkpoints"] = checkpoints

    # Save results
    if save_dir:
        save_path = Path(save_dir)
        # Save log as JSON (without checkpoints — those are torch tensors)
        log_data = {k: v for k, v in result.items() if k != "checkpoints"}
        with open(save_path / "log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        # Save checkpoints separately
        if checkpoints:
            torch.save(checkpoints, save_path / "checkpoints.pt")
        # Save final model
        torch.save(model.state_dict(), save_path / "model_final.pt")

    return result
