"""Visualization functions for grokking experiment results."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


def plot_training_curves(log: dict, config: dict, save_path: str | None = None):
    """Plot training/test accuracy and loss vs. epoch.

    Shows the four grokking phases:
    I   - Memorization (train acc -> 100%)
    II  - Plateau (train loss ~ 0, test loss stationary)
    III - Circuit formation (internal restructuring)
    IV  - Generalization (test acc -> 100%)
    """
    epochs = log["epoch"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Accuracy
    ax1.plot(epochs, log["train_acc"], label="Train accuracy", color="tab:blue")
    ax1.plot(epochs, log["test_acc"], label="Test accuracy", color="tab:orange")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, log["train_loss"], label="Train loss", color="tab:blue")
    ax2.plot(epochs, log["test_loss"], label="Test loss", color="tab:orange")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    alpha = config.get("alpha", "?")
    wd = config.get("weight_decay", "?")
    fig.suptitle(f"Training Curves (alpha={alpha}, lambda={wd})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_order_parameters(log: dict, config: dict, save_path: str | None = None):
    """Plot all 5 order parameters vs. epoch."""
    epochs = log["epoch"]

    param_names = [
        ("rqi", "RQI"),
        ("effective_rank", "Effective Rank"),
        ("participation_ratio", "Participation Ratio"),
        ("kernel_alignment", "Kernel Alignment"),
        ("snr", "SNR"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    for ax, (key, label) in zip(axes, param_names):
        if key not in log:
            ax.text(0.5, 0.5, f"{label}: not computed", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        vals = log[key]
        ax.plot(epochs[:len(vals)], vals, color="tab:green", linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if key == "snr":
            ax.set_yscale("log")
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="SNR=1")
            ax.legend()

    axes[-1].set_xlabel("Epoch")

    alpha = config.get("alpha", "?")
    wd = config.get("weight_decay", "?")
    fig.suptitle(f"Order Parameters (alpha={alpha}, lambda={wd})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_transition(
    log: dict,
    config: dict,
    t_grok: int | None = None,
    window: int = 5000,
    save_path: str | None = None,
):
    """Zoomed view around the grokking instant with phase transition markers."""
    if t_grok is None:
        print("No grokking detected — skipping phase transition plot.")
        return None

    epochs = np.array(log["epoch"])
    mask = (epochs >= t_grok - window) & (epochs <= t_grok + window)
    if mask.sum() < 2:
        print("Not enough data points around grokking instant.")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Accuracy
    ax = axes[0]
    ax.plot(epochs[mask], np.array(log["train_acc"])[mask], label="Train acc")
    ax.plot(epochs[mask], np.array(log["test_acc"])[mask], label="Test acc")
    ax.axvline(x=t_grok, color="red", linestyle="--", alpha=0.7, label=f"t_grok={t_grok}")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RQI and Kernel Alignment
    ax = axes[1]
    if "rqi" in log:
        rqi = np.array(log["rqi"])
        ax.plot(epochs[mask][:len(rqi[mask])], rqi[mask], label="RQI")
    if "kernel_alignment" in log:
        ka = np.array(log["kernel_alignment"])
        ax.plot(epochs[mask][:len(ka[mask])], ka[mask], label="Kernel Alignment")
    ax.axvline(x=t_grok, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SNR
    ax = axes[2]
    if "snr" in log:
        snr = np.array(log["snr"])
        ax.plot(epochs[mask][:len(snr[mask])], snr[mask], label="SNR", color="tab:green")
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=t_grok, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("SNR")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase Transition Detail (t_grok={t_grok})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_diagram(sweep_results: list[dict], save_path: str | None = None):
    """2D heatmap of t_grok/t_mem over (alpha, lambda) grid.

    Args:
        sweep_results: List of dicts, each with keys: alpha, weight_decay,
                       t_mem, t_grok.
    """
    alphas = sorted(set(r["alpha"] for r in sweep_results))
    lambdas = sorted(set(r["weight_decay"] for r in sweep_results))

    ratio_grid = np.full((len(lambdas), len(alphas)), np.nan)

    for r in sweep_results:
        i = lambdas.index(r["weight_decay"])
        j = alphas.index(r["alpha"])
        t_mem = r.get("t_mem")
        t_grok = r.get("t_grok")
        if t_mem and t_grok and t_mem > 0:
            ratio_grid[i, j] = t_grok / t_mem
        else:
            ratio_grid[i, j] = np.inf

    fig, ax = plt.subplots(figsize=(8, 6))

    # Replace inf with a large finite value for plotting
    plot_grid = ratio_grid.copy()
    max_finite = np.nanmax(plot_grid[np.isfinite(plot_grid)]) if np.any(np.isfinite(plot_grid)) else 100
    plot_grid[~np.isfinite(plot_grid)] = max_finite * 2

    im = ax.imshow(
        plot_grid,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn_r",
        norm=mcolors.LogNorm(vmin=1, vmax=max_finite * 2),
    )
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a}" for a in alphas])
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels([f"{l}" for l in lambdas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("weight decay (lambda)")
    ax.set_title("Phase Diagram: t_grok / t_mem")

    # Annotate cells
    for i in range(len(lambdas)):
        for j in range(len(alphas)):
            val = ratio_grid[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "inf", ha="center", va="center", fontsize=9, color="white")

    fig.colorbar(im, ax=ax, label="t_grok / t_mem")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_svd_evolution(
    checkpoints: dict[int, dict],
    layer_key: str = "fc1.weight",
    save_path: str | None = None,
):
    """Visualize singular value spectrum at key training snapshots.

    Shows emergence of signal spikes from bulk noise (BBP transition).

    Args:
        checkpoints: Dict mapping epoch -> state_dict.
        layer_key: Which weight matrix to decompose.
    """
    epochs_sorted = sorted(checkpoints.keys())
    n = len(epochs_sorted)
    if n == 0:
        print("No checkpoints provided.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for epoch, color in zip(epochs_sorted, colors):
        W = checkpoints[epoch][layer_key]
        if isinstance(W, torch.Tensor):
            S = torch.linalg.svdvals(W.float()).cpu().numpy()
        else:
            S = np.linalg.svd(W, compute_uv=False)
        ax.plot(range(len(S)), S, color=color, label=f"Epoch {epoch}", linewidth=1.5)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Weight Matrix SVD Evolution (W1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
