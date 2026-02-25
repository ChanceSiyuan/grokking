"""Hyperparameter sweep over (alpha, weight_decay) for phase diagram."""

import json
from pathlib import Path

from .train import train


DEFAULT_ALPHAS = [0.0, 0.1, 0.25, 0.5]
DEFAULT_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]


def run_sweep(
    alphas: list[float] = DEFAULT_ALPHAS,
    lambdas: list[float] = DEFAULT_LAMBDAS,
    save_dir: str = "results/sweep",
    n_epochs: int = 100_000,
    **train_kwargs,
) -> list[dict]:
    """Run sequential sweep over (alpha, lambda) grid.

    Args:
        alphas: List of alpha values to sweep.
        lambdas: List of weight_decay values to sweep.
        save_dir: Base directory for results.
        n_epochs: Max epochs per run.
        **train_kwargs: Additional keyword args passed to train().

    Returns:
        List of summary dicts for each run.
    """
    base_dir = Path(save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(alphas) * len(lambdas)
    run_idx = 0

    for alpha in alphas:
        for wd in lambdas:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total}: alpha={alpha}, lambda={wd}")
            print(f"{'='*60}")

            run_dir = base_dir / f"alpha{alpha}_wd{wd}"

            result = train(
                alpha=alpha,
                weight_decay=wd,
                n_epochs=n_epochs,
                save_dir=str(run_dir),
                compute_order_params=True,
                **train_kwargs,
            )

            summary = {
                "alpha": alpha,
                "weight_decay": wd,
                "t_mem": result["t_mem"],
                "t_grok": result["t_grok"],
                "elapsed_seconds": result["elapsed_seconds"],
                "final_train_acc": result["log"]["train_acc"][-1],
                "final_test_acc": result["log"]["test_acc"][-1],
            }
            results.append(summary)

            print(f"  t_mem={summary['t_mem']}, t_grok={summary['t_grok']}")
            print(f"  Final train_acc={summary['final_train_acc']:.4f}, "
                  f"test_acc={summary['final_test_acc']:.4f}")

    # Save sweep summary
    with open(base_dir / "sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
