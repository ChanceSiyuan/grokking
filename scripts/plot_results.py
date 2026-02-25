"""Load saved experiment logs and regenerate all figures."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from grokking.viz import (
    plot_training_curves,
    plot_order_parameters,
    plot_phase_transition,
    plot_svd_evolution,
    plot_phase_diagram,
)


def plot_single(result_dir: str):
    """Regenerate plots for a single experiment."""
    result_dir = Path(result_dir)

    with open(result_dir / "log.json") as f:
        data = json.load(f)

    log = data["log"]
    config = data["config"]
    t_grok = data.get("t_grok")

    plot_training_curves(log, config, save_path=str(result_dir / "training_curves.png"))
    print(f"  Saved training_curves.png")

    if "rqi" in log:
        plot_order_parameters(log, config, save_path=str(result_dir / "order_parameters.png"))
        print(f"  Saved order_parameters.png")

        plot_phase_transition(log, config, t_grok=t_grok,
                              save_path=str(result_dir / "phase_transition.png"))
        print(f"  Saved phase_transition.png")

    ckpt_path = result_dir / "checkpoints.pt"
    if ckpt_path.exists():
        checkpoints = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        plot_svd_evolution(checkpoints, save_path=str(result_dir / "svd_evolution.png"))
        print(f"  Saved svd_evolution.png")


def plot_sweep(sweep_dir: str):
    """Regenerate phase diagram from sweep results."""
    sweep_dir = Path(sweep_dir)

    with open(sweep_dir / "sweep_summary.json") as f:
        results = json.load(f)

    plot_phase_diagram(results, save_path=str(sweep_dir / "phase_diagram.png"))
    print(f"  Saved phase_diagram.png")


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved results")
    parser.add_argument("path", type=str, help="Path to result directory")
    parser.add_argument("--type", choices=["single", "sweep"], default="single",
                        help="Type of results to plot")
    args = parser.parse_args()

    print(f"Plotting results from {args.path}")
    if args.type == "single":
        plot_single(args.path)
    else:
        plot_sweep(args.path)
    print("Done.")


if __name__ == "__main__":
    main()
