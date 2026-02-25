"""2-layer MLP with muP initialization for grokking experiments."""

import math
import torch
import torch.nn as nn


class GrokMLP(nn.Module):
    """2-layer MLP with muP-style alpha scaling.

    Architecture: Linear(2p, N) -> scale by N^{-alpha} -> ReLU -> Linear(N, p)

    The scale parameter alpha controls the training regime via
    f(x) = W2 @ relu(N^{-alpha} * W1 @ x):

        alpha = 0   -> rich/feature-learning regime (grokking possible)
        alpha = 0.5 -> lazy/NTK regime (memorization only)

    The forward-pass scaling by N^{-alpha} creates a timescale separation:
        tau_features ~ N^{2*alpha} / eta
    At alpha=0 features learn at the same rate as readout; at alpha=0.5
    features are N times slower, suppressing feature learning.
    """

    def __init__(self, p: int = 97, width: int = 512, alpha: float = 0.0):
        super().__init__()
        self.p = p
        self.width = width
        self.alpha = alpha

        self.fc1 = nn.Linear(2 * p, width, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(width, p, bias=True)

        self._mup_scale = width ** (-alpha) if alpha > 0 else 1.0

        self._init_weights()

    def _init_weights(self):
        N = self.width
        # W1 ~ N(0, 1/fan_in)
        nn.init.normal_(self.fc1.weight, std=1.0 / math.sqrt(2 * self.p))
        nn.init.zeros_(self.fc1.bias)
        # W2 ~ N(0, 1/N)
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(N))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = h * self._mup_scale
        h = self.relu(h)
        return self.fc2(h)
