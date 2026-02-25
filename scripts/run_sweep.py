"""Run the full (alpha, lambda) hyperparameter sweep."""

import argparse
import sys
sys.path.insert(0, "src")

from grokking.sweep import run_sweep
from grokking.viz import plot_phase_diagram


def main():
    parser = argparse.ArgumentParser(description="Run (alpha, lambda) sweep")
    parser.add_argument("--n-epochs", type=int, default=100_000, help="Max epochs per run")
    parser.add_argument("--save-dir", type=str, default="results/sweep", help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_sweep(
        save_dir=args.save_dir,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Sweep complete! Generating phase diagram...")
    plot_phase_diagram(results, save_path=f"{args.save_dir}/phase_diagram.png")
    print(f"Phase diagram saved to {args.save_dir}/phase_diagram.png")

    # Print summary table
    print("\nSummary:")
    print(f"{'alpha':>6} {'lambda':>8} {'t_mem':>8} {'t_grok':>8} {'ratio':>8} {'test_acc':>9}")
    print("-" * 55)
    for r in results:
        t_mem = r["t_mem"] or "N/A"
        t_grok = r["t_grok"] or "N/A"
        if isinstance(t_mem, int) and isinstance(t_grok, int) and t_mem > 0:
            ratio = f"{t_grok / t_mem:.1f}"
        else:
            ratio = "inf"
        print(f"{r['alpha']:>6.2f} {r['weight_decay']:>8.4f} {str(t_mem):>8} "
              f"{str(t_grok):>8} {ratio:>8} {r['final_test_acc']:>9.4f}")


if __name__ == "__main__":
    main()
