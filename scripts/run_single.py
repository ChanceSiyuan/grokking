"""Run a single grokking experiment with configurable hyperparameters."""

import argparse
import sys
sys.path.insert(0, "src")

from grokking.train import train
from grokking.viz import (
    plot_training_curves,
    plot_order_parameters,
    plot_phase_transition,
    plot_svd_evolution,
)


def main():
    parser = argparse.ArgumentParser(description="Run a single grokking experiment")
    parser.add_argument("--p", type=int, default=97, help="Prime modulus")
    parser.add_argument("--width", type=int, default=512, help="Hidden layer width")
    parser.add_argument("--alpha", type=float, default=0.0, help="muP scale parameter (0=rich, 0.5=lazy)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--train-fraction", type=float, default=0.3, help="Training data fraction")
    parser.add_argument("--n-epochs", type=int, default=100_000, help="Max epochs")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval")
    parser.add_argument("--save-dir", type=str, default="results/single", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-order-params", action="store_true", help="Skip order parameter computation")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    # Define checkpoint epochs (will be refined after t_mem is known)
    # For now, checkpoint at regular intervals
    checkpoint_epochs = [0, 500, 1000, 5000, 10000, 25000, 50000, 75000, 99999]

    result = train(
        p=args.p,
        width=args.width,
        alpha=args.alpha,
        weight_decay=args.weight_decay,
        lr=args.lr,
        train_fraction=args.train_fraction,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        checkpoint_epochs=checkpoint_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
        compute_order_params=not args.no_order_params,
    )

    print(f"\nResults:")
    print(f"  t_mem  = {result['t_mem']}")
    print(f"  t_grok = {result['t_grok']}")
    print(f"  Final train acc: {result['log']['train_acc'][-1]:.4f}")
    print(f"  Final test acc:  {result['log']['test_acc'][-1]:.4f}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_training_curves(
            result["log"], result["config"],
            save_path=f"{args.save_dir}/training_curves.png",
        )
        if not args.no_order_params:
            plot_order_parameters(
                result["log"], result["config"],
                save_path=f"{args.save_dir}/order_parameters.png",
            )
            plot_phase_transition(
                result["log"], result["config"],
                t_grok=result["t_grok"],
                save_path=f"{args.save_dir}/phase_transition.png",
            )
        if result.get("checkpoints"):
            plot_svd_evolution(
                result["checkpoints"],
                save_path=f"{args.save_dir}/svd_evolution.png",
            )
        print(f"Plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
