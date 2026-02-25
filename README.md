# Grokking: Empirical Reproduction on Modular Arithmetic

Reproducing the **grokking** phenomenon on modular addition tasks, tracking phase transition diagnostics with five order parameters.

Grokking is a phenomenon where neural networks first memorize training data, then — after a long delay — suddenly generalize to unseen data. This project implements the full experimental infrastructure described in [Issue #4](https://github.com/ChanceSiyuan/grokking/issues/4).

**📖 [Read the full notes →](https://chancesiyuan.github.io/grokking/)**

- [Survey: Grokking as a Phase Transition](https://chancesiyuan.github.io/grokking/index.html) — statistical physics and mean field theory perspective
- [Analytical Proof of Grokking Delay](https://chancesiyuan.github.io/grokking/grokking-delay-proof.html) — timescale separation and Kramers barrier analysis

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quick Start

### Single Experiment

```bash
uv run python scripts/run_single.py \
    --alpha 0.0 \
    --weight-decay 1.0 \
    --n-epochs 30000 \
    --save-dir results/single
```

### Hyperparameter Sweep

Sweep over α ∈ {0, 0.1, 0.25, 0.5} × λ ∈ {1e-3, 1e-2, 1e-1, 1} (16 runs):

```bash
uv run python scripts/run_sweep.py \
    --n-epochs 100000 \
    --save-dir results/sweep
```

### Regenerate Plots

```bash
uv run python scripts/plot_results.py results/single --type single
uv run python scripts/plot_results.py results/sweep --type sweep
```

## Architecture

- **Model:** 2-layer MLP (Linear → ReLU → Linear), width N=512
- **Task:** Modular addition a + b (mod 97)
- **Data:** 30% of all 97² = 9409 pairs for training, 70% for test
- **Optimizer:** AdamW, lr=1e-3
- **μP scaling:** Forward pass scaled by N^{-α}, controlling lazy (α=0.5) vs rich (α=0) regime

## Order Parameters

Five diagnostics tracked during training:

| Parameter | Description |
|-----------|-------------|
| **RQI** | Representation Quality Index — embedding geometry regularity |
| **Effective Rank** | Singular value entropy of weight matrix |
| **Participation Ratio** | Active modes in hidden activation covariance |
| **Kernel Alignment** | Alignment between empirical NTK and ideal task kernel |
| **SNR** | Signal-to-noise ratio in Fourier basis |

## Project Structure

```
src/grokking/
├── data.py       # Modular arithmetic dataset
├── model.py      # 2-layer MLP with μP scaling
├── metrics.py    # Order parameter computation
├── train.py      # Training loop
├── viz.py        # Plotting functions
└── sweep.py      # Hyperparameter sweep
scripts/
├── run_single.py   # Run one experiment
├── run_sweep.py    # Run (α, λ) grid
└── plot_results.py # Regenerate figures
```

## CLI Options

```
usage: run_single.py [-h] [--p P] [--width WIDTH] [--alpha ALPHA]
                     [--weight-decay WEIGHT_DECAY] [--lr LR]
                     [--train-fraction TRAIN_FRACTION] [--n-epochs N_EPOCHS]
                     [--log-every LOG_EVERY] [--save-dir SAVE_DIR]
                     [--seed SEED] [--no-order-params] [--no-plots]
```

## References

- [Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.](https://arxiv.org/abs/2201.02177)
- [Nanda et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability.](https://arxiv.org/abs/2301.05217)
- [Liu et al. (2022). Omnigrok: Grokking Beyond Algorithmic Data.](https://arxiv.org/abs/2210.01117)
- [Kumar et al. (2024). Grokking as the Transition from Lazy to Rich Training Dynamics.](https://arxiv.org/abs/2310.06110)

## License

MIT
