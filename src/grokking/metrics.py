"""Order parameter computation for grokking phase transition diagnostics."""

import math
import torch
import numpy as np


def compute_effective_rank(W: torch.Tensor) -> float:
    """Effective rank of weight matrix via singular value entropy.

    erank(W) = exp(-sum_i p_i log p_i)
    where p_i = sigma_i^2 / sum_j sigma_j^2.
    """
    S = torch.linalg.svdvals(W)
    S2 = S ** 2
    total = S2.sum()
    if total < 1e-12:
        return 0.0
    p = S2 / total
    # Avoid log(0)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def compute_participation_ratio(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> float:
    """Participation ratio of hidden-layer activation covariance.

    PR = tr(C)^2 / tr(C^2) where C is the covariance of hidden activations.
    """
    with torch.no_grad():
        h = model.fc1(x) * model._mup_scale
        h = model.relu(h)  # (n_samples, width)

    # Center activations
    h = h - h.mean(dim=0, keepdim=True)
    n = h.shape[0]
    # Covariance: C = h^T h / n
    C = (h.T @ h) / n

    tr_C = torch.trace(C)
    tr_C2 = (C * C).sum()  # tr(C^2) = ||C||_F^2

    if tr_C2 < 1e-12:
        return 0.0
    return (tr_C ** 2 / tr_C2).item()


def _build_fourier_projections(p: int, device: str = "cpu"):
    """Build signal and noise projection matrices for Fourier SNR.

    For modular addition mod p, the signal subspace is spanned by the
    discrete Fourier basis vectors at all non-zero frequencies k=1,...,p-1.
    The network needs to learn Fourier components to solve the task.

    We project the first-layer weights (which embed the inputs) onto the
    Fourier basis for Z_p. The signal subspace consists of the Fourier
    modes; the noise subspace is the complement (DC component / uniform).
    """
    # DFT matrix for Z_p: F[k, j] = exp(2*pi*i*k*j / p) / sqrt(p)
    k = torch.arange(p, device=device, dtype=torch.float64)
    j = torch.arange(p, device=device, dtype=torch.float64)
    angles = 2 * math.pi * k.unsqueeze(1) * j.unsqueeze(0) / p
    # Real Fourier basis: cos and sin components for k=1,...,(p-1)//2
    # k=0 is the DC (noise/mean) component
    basis_vectors = []
    for freq in range(1, (p + 1) // 2):
        cos_vec = torch.cos(angles[freq])  # shape (p,)
        sin_vec = torch.sin(angles[freq])
        # Normalize
        cos_vec = cos_vec / cos_vec.norm()
        sin_vec = sin_vec / sin_vec.norm()
        basis_vectors.append(cos_vec)
        basis_vectors.append(sin_vec)

    if len(basis_vectors) == 0:
        V_signal = torch.zeros(p, 0, device=device, dtype=torch.float64)
    else:
        V_signal = torch.stack(basis_vectors, dim=1)  # (p, 2*num_freqs)

    # Signal projection: Pi_S = V V^T
    Pi_S = V_signal @ V_signal.T  # (p, p)
    # Noise projection: complement
    Pi_N = torch.eye(p, device=device, dtype=torch.float64) - Pi_S

    return Pi_S.float(), Pi_N.float()


def compute_snr(
    model: torch.nn.Module,
    p: int,
    Pi_S: torch.Tensor,
    Pi_N: torch.Tensor,
) -> float:
    """Signal-to-noise ratio of first-layer weights in Fourier basis.

    SNR = ||Pi_S @ W||_F^2 / ||Pi_N @ W||_F^2

    W is split into two blocks of p rows: embeddings for a and for b.
    We compute SNR for each block and average.
    """
    with torch.no_grad():
        W = model.fc1.weight.data  # (width, 2p)
        # Transpose so rows are input dimensions: (2p, width)
        W = W.T

        snr_total = 0.0
        for block in range(2):
            W_block = W[block * p : (block + 1) * p, :]  # (p, width)
            E_S = (Pi_S @ W_block).norm() ** 2
            E_N = (Pi_N @ W_block).norm() ** 2
            if E_N < 1e-12:
                snr_total += 1e6  # effectively infinite
            else:
                snr_total += (E_S / E_N).item()

    return snr_total / 2.0


def compute_rqi(
    model: torch.nn.Module,
    p: int,
    n_samples: int = 2000,
    threshold: float = 0.5,
    seed: int = 0,
) -> float:
    """Representation Quality Index.

    Fraction of sampled quadruples (a,b,c,d) with a+b ≡ c+d (mod p) where
    ||e_a + e_b - e_c - e_d|| < threshold.

    Embeddings are the rows of the first-layer weight matrix corresponding
    to each input token.
    """
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        W = model.fc1.weight.data.T  # (2p, width)
        # Embeddings for token a: rows 0..p-1
        # Embeddings for token b: rows p..2p-1
        E_a = W[:p]  # (p, width)
        E_b = W[p:]  # (p, width)

    count_pass = 0
    for _ in range(n_samples):
        a = rng.randint(0, p)
        b = rng.randint(0, p)
        target = (a + b) % p
        # Pick c randomly, then d = (target - c) mod p
        c = rng.randint(0, p)
        d = (target - c) % p
        # Skip trivial case (a,b) == (c,d)
        if a == c and b == d:
            continue
        diff = E_a[a] + E_b[b] - E_a[c] - E_b[d]
        if diff.norm().item() < threshold:
            count_pass += 1

    return count_pass / n_samples


def _build_ideal_kernel(labels: torch.Tensor) -> torch.Tensor:
    """Build ideal task kernel: K*[i,j] = 1 if y_i == y_j, else 0."""
    return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()


def compute_kernel_alignment(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_subsample: int = 500,
    seed: int = 0,
) -> float:
    """Kernel alignment between empirical NTK and ideal task kernel.

    A(K_t, K*) = <K_t, K*>_F / (||K_t||_F ||K*||_F)

    Uses a subsample for efficiency since full NTK is O(n^2) in memory.
    """
    n = x.shape[0]
    if n > n_subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n_subsample, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Compute empirical NTK via Jacobian
    m = x_sub.shape[0]
    x_sub = x_sub.detach().requires_grad_(False)

    # Collect gradients for each sample
    params = [p for p in model.parameters() if p.requires_grad]
    grads_list = []

    for i in range(m):
        model.zero_grad()
        out = model(x_sub[i : i + 1])  # (1, p)
        # Use the sum of logits as the scalar output for NTK
        out.sum().backward()
        grad_vec = torch.cat([p.grad.flatten() for p in params])
        grads_list.append(grad_vec)

    J = torch.stack(grads_list)  # (m, n_params)
    K_t = J @ J.T  # (m, m)

    K_star = _build_ideal_kernel(y_sub)

    # Frobenius inner product
    num = (K_t * K_star).sum()
    denom = K_t.norm() * K_star.norm()
    if denom < 1e-12:
        return 0.0
    return (num / denom).item()


class OrderParameterTracker:
    """Efficiently track all 5 order parameters during training."""

    def __init__(self, model, p: int, device: str = "cpu"):
        self.model = model
        self.p = p
        self.device = device
        self.Pi_S, self.Pi_N = _build_fourier_projections(p, device)

    def compute_all(
        self,
        x_all: torch.Tensor,
        y_all: torch.Tensor,
        rqi_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute all 5 order parameters.

        Args:
            x_all: Full input tensor (for participation ratio and kernel alignment).
            y_all: Full label tensor (for kernel alignment).
            rqi_threshold: Distance threshold for RQI.

        Returns:
            Dict with keys: rqi, effective_rank, participation_ratio,
                           kernel_alignment, snr.
        """
        W1 = self.model.fc1.weight.data

        return {
            "rqi": compute_rqi(self.model, self.p, threshold=rqi_threshold),
            "effective_rank": compute_effective_rank(W1),
            "participation_ratio": compute_participation_ratio(self.model, x_all),
            "kernel_alignment": compute_kernel_alignment(
                self.model, x_all, y_all
            ),
            "snr": compute_snr(self.model, self.p, self.Pi_S, self.Pi_N),
        }
