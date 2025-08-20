#!/usr/bin/env python
"""
Memory-safe, numerically stable MI-like scores (CKA-style) between consecutive tokens.

Key feature: STREAMING RBF CKA that avoids building [H,H] kernels.
- Works per token-pair in tiles of size 'chunk_size' along the second axis.
- O(H * chunk_size) memory instead of O(H^2).
- Optional hidden-dim subsampling for extra speed.

API examples
-----------
# activations: [T, L, H] tensor
from mi_fast_stream import compute_mi

# Fastest & tiny memory: linear CKA on last layer
transitions = compute_mi(activations, method="linear")               # [T-1]

# Memory-safe RBF CKA on last layer (streaming)
transitions = compute_mi(activations, method="rbf",
                         rbf_mode="stream", sigma=50.0,
                         chunk_size=2048, use_double=False)          # [T-1]

# With hidden unit subsampling (e.g., 2048 dims)
transitions = compute_mi(activations, method="rbf",
                         rbf_mode="stream", sigma=50.0,
                         subsample=2048, chunk_size=2048)

# All layers (linear CKA), returns [T-1, L]
transitions_all = compute_mi(activations, layer="all", method="linear")
"""

from __future__ import annotations
from typing import Literal, Optional, Union
import torch

# ======================================================================================
# Utilities
# ======================================================================================

def _work_dtype(use_double: bool) -> torch.dtype:
    return torch.float64 if use_double else torch.float32

def _to_work(x: torch.Tensor, use_double: bool) -> torch.Tensor:
    """Cast to float32/float64 to avoid bf16/half CPU/GPU op limitations."""
    return x.to(dtype=_work_dtype(use_double))

def _pick_layer_slices(
    activations: torch.Tensor,
    layer: Optional[int | Literal["last", "all"]] = "last",
) -> torch.Tensor:
    """
    activations: [T, L, H]
    Returns:
      - layer='last' or int: [T, H]
      - layer='all':        [T, L, H]
    """
    if layer in (None, "last"):
        return activations[:, -1, :]
    if layer == "all":
        return activations
    if isinstance(layer, int):
        return activations[:, layer, :]
    raise ValueError("layer must be 'last', 'all', or an int")

def _maybe_subsample_hidden(
    X: torch.Tensor,
    subsample: Optional[Union[int, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Subsample hidden units for efficiency.
    - subsample int: keep exactly that many units (<= H)
    - subsample float in (0,1]: keep that fraction of units
    Returns (X[:, idx], idx) where idx=None if no subsampling.
    """
    if subsample is None:
        return X, None
    T = X.shape[0]
    H = X.shape[-1]
    if isinstance(subsample, float):
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample float must be in (0, 1].")
        k = max(1, int(round(subsample * H)))
    else:
        k = int(subsample)
        if not (1 <= k <= H):
            raise ValueError(f"subsample int must be in [1, H={H}]")
    idx = torch.randperm(H, device=X.device, generator=generator)[:k]
    if X.ndim == 2:   # [T, H]
        return X[:, idx], idx
    else:             # [T, L, H]
        return X[..., idx], idx

# ======================================================================================
# Linear CKA (fastest; tiny memory)
# ======================================================================================

def compute_mi_linear_cka(
    activations: torch.Tensor,
    *,
    layer: Optional[int | Literal["last", "all"]] = "last",
    eps: float = 1e-12,
    use_double: bool = False,
    subsample: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """
    Vectorized linear CKA for consecutive tokens.
    - activations: [T, L, H]
    - returns [T-1] (single layer) or [T-1, L] (all layers)
    """
    X = _pick_layer_slices(activations, layer=layer)  # [T,H] or [T,L,H]
    if X.ndim == 2:
        X, _ = _maybe_subsample_hidden(X, subsample=subsample)
        A = _to_work(X[:-1, :], use_double)
        B = _to_work(X[1:,  :], use_double)
        A = A - A.mean(dim=1, keepdim=True)
        B = B - B.mean(dim=1, keepdim=True)
        num = (A * B).sum(dim=1)
        den = A.norm(dim=1) * B.norm(dim=1) + torch.tensor(eps, dtype=A.dtype, device=A.device)
        return (num / den).pow(2)
    else:
        if subsample is not None:
            X, _ = _maybe_subsample_hidden(X, subsample=subsample)
        T, L, H = X.shape
        A = _to_work(X[:-1], use_double)   # [T-1, L, H]
        B = _to_work(X[1:],  use_double)
        A = A - A.mean(dim=2, keepdim=True)
        B = B - B.mean(dim=2, keepdim=True)
        num = (A * B).sum(dim=2)                              # [T-1, L]
        den = (A.norm(dim=2) * B.norm(dim=2)) + torch.tensor(
            eps, dtype=A.dtype, device=A.device
        )
        return (num / den).pow(2)

# ======================================================================================
# RBF (Gaussian) CKA — streaming, O(H*chunk) memory (safe)
# ======================================================================================

@torch.no_grad()
def _rbf_cka_pair_streaming(
    x: torch.Tensor,  # [H]
    y: torch.Tensor,  # [H]
    *,
    sigma: float = 50.0,
    eps: float = 1e-12,
    use_double: bool = False,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Compute CKA(Kx_c, Ky_c) for TWO 1D vectors x,y without materializing HxH.
    Tiles over the second index in blocks of 'chunk_size'.

    Returns a scalar tensor on the same device.
    """
    device = x.device
    dtype  = _work_dtype(use_double)

    x = _to_work(x.view(-1), use_double)  # [H]
    y = _to_work(y.view(-1), use_double)  # [H]
    H = x.numel()
    chunk = max(1, min(chunk_size, H))

    inv_two_sigma2 = torch.tensor(1.0 / (2.0 * (sigma ** 2)), dtype=dtype, device=device)

    # Accumulators
    row_sum_x = torch.zeros(H, dtype=dtype, device=device)
    row_sum_y = torch.zeros(H, dtype=dtype, device=device)
    sum_x  = torch.tensor(0.0, dtype=dtype, device=device)
    sum_y  = torch.tensor(0.0, dtype=dtype, device=device)
    sum_xy = torch.tensor(0.0, dtype=dtype, device=device)  # <Kx, Ky>
    sum_xx = torch.tensor(0.0, dtype=dtype, device=device)  # <Kx, Kx>
    sum_yy = torch.tensor(0.0, dtype=dtype, device=device)  # <Ky, Ky>

    # Process tiles along the "j" dimension
    for j0 in range(0, H, chunk):
        j1 = min(H, j0 + chunk)
        xj = x[j0:j1]                    # [C]
        yj = y[j0:j1]                    # [C]

        # [H, C] differences (streaming block)
        diff_x = x.unsqueeze(1) - xj.unsqueeze(0)
        diff_y = y.unsqueeze(1) - yj.unsqueeze(0)

        Kx_blk = torch.exp(-(diff_x.pow(2)) * inv_two_sigma2)   # [H, C]
        Ky_blk = torch.exp(-(diff_y.pow(2)) * inv_two_sigma2)   # [H, C]

        # Row sums and totals
        row_sum_x += Kx_blk.sum(dim=1)
        row_sum_y += Ky_blk.sum(dim=1)
        sum_x     += Kx_blk.sum()
        sum_y     += Ky_blk.sum()

        # Frobenius inner products
        sum_xy    += (Kx_blk * Ky_blk).sum()     # <Kx,Ky> over block
        sum_xx    += (Kx_blk.square()).sum()     # <Kx,Kx> over block
        sum_yy    += (Ky_blk.square()).sum()     # <Ky,Ky> over block

        # Free the block ASAP
        del diff_x, diff_y, Kx_blk, Ky_blk

    # Centering corrections via HKH formula:
    # <Kc,Lc> = <K,L> - (2/H) rK·rL + (1/H^2) sumK * sumL
    # ||Kc||^2 = <K,K> - (2/H) rK·rK + (1/H^2) sumK^2
    invH  = torch.tensor(1.0 / H, dtype=dtype, device=device)
    invH2 = torch.tensor(1.0 / (H * H), dtype=dtype, device=device)

    rKrL = (row_sum_x * row_sum_y).sum()
    num  = sum_xy - (2.0 * invH) * rKrL + invH2 * (sum_x * sum_y)

    rKrK = (row_sum_x.square()).sum()
    rLrL = (row_sum_y.square()).sum()
    nk   = sum_xx - (2.0 * invH) * rKrK + invH2 * (sum_x * sum_x)
    nl   = sum_yy - (2.0 * invH) * rLrL + invH2 * (sum_y * sum_y)

    # Guard tiny negatives from numerical noise
    nk = torch.clamp(nk, min=0.0)
    nl = torch.clamp(nl, min=0.0)

    denom = torch.sqrt(nk) * torch.sqrt(nl) + torch.tensor(eps, dtype=dtype, device=device)
    val = num / denom
    return val.clamp(min=0.0, max=1.0)

def compute_mi_rbf_cka_streaming(
    activations: torch.Tensor,
    *,
    layer: Optional[int | Literal["last", "all"]] = "last",
    sigma: float = 50.0,
    eps: float = 1e-12,
    use_double: bool = False,
    chunk_size: int = 2048,
    subsample: Optional[Union[int, float]] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Streaming RBF CKA over consecutive tokens (safe for big H).
    Returns [T-1] for single layer, or [T-1, L] for 'all'.
    """
    if layer == "all":
        # Process each layer independently (loop over L)
        X = _pick_layer_slices(activations, layer="all")  # [T, L, H]
        if subsample is not None:
            X, idx = _maybe_subsample_hidden(X, subsample=subsample)
        T, L, H = X.shape
        out = torch.empty(T - 1, L, dtype=_work_dtype(use_double), device=X.device)
        it = range(L)
        if show_progress:
            try:
                from tqdm import tqdm as _tqdm
                it = _tqdm(it, desc="RBF CKA layers")
            except Exception:
                pass
        for l in it:
            xl = X[:-1, l, :]  # [T-1, H]
            yl = X[1:,  l, :]  # [T-1, H]
            vals = []
            for i in range(T - 1):
                vals.append(_rbf_cka_pair_streaming(
                    xl[i], yl[i],
                    sigma=sigma, eps=eps, use_double=use_double, chunk_size=chunk_size
                ))
            out[:, l] = torch.stack(vals)
        return out

    # Single layer ("last" or specific int)
    X = _pick_layer_slices(activations, layer=layer)  # [T, H]
    if subsample is not None:
        X, idx = _maybe_subsample_hidden(X, subsample=subsample)
    T, H = X.shape
    vals = []
    it = range(T - 1)
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
            it = _tqdm(it, desc="RBF CKA (stream)")
        except Exception:
            pass
    for i in it:
        vals.append(_rbf_cka_pair_streaming(
            X[i], X[i + 1],
            sigma=sigma, eps=eps, use_double=use_double, chunk_size=chunk_size
        ))
    return torch.stack(vals)  # [T-1]

# ======================================================================================
# (Optional) RBF CKA — batched (OOM-prone for large H; kept for reference)
# ======================================================================================

def compute_mi_rbf_cka_batched(
    activations: torch.Tensor,
    *,
    layer: Optional[int | Literal["last", "all"]] = "last",
    sigma: float = 50.0,
    eps: float = 1e-12,
    use_double: bool = False,
) -> torch.Tensor:
    """
    Original batched RBF CKA (builds [B,H,H] kernels) — use only for small H.
    """
    def _center_gram_batched(K: torch.Tensor) -> torch.Tensor:
        mean_row = K.mean(dim=2, keepdim=True)
        mean_col = K.mean(dim=1, keepdim=True)
        mean_all = K.mean(dim=(1, 2), keepdim=True)
        return K - mean_row - mean_col + mean_all

    def _rbf_gram_batched(X: torch.Tensor, sigma: float, use_double: bool) -> torch.Tensor:
        X = _to_work(X, use_double)
        diff = X.unsqueeze(2) - X.unsqueeze(1)  # [B, H, H]  <-- OOM risk!
        D2 = diff.pow(2)
        sigma2 = torch.tensor(sigma, dtype=X.dtype, device=X.device).pow(2)
        return torch.exp(-D2 / (2.0 * sigma2))

    X = _pick_layer_slices(activations, layer=layer)
    if X.ndim == 2:
        A = X[:-1, :]
        B = X[1:,  :]
        Kx = _rbf_gram_batched(A, sigma, use_double)
        Ky = _rbf_gram_batched(B, sigma, use_double)
        Kx_c = _center_gram_batched(Kx)
        Ky_c = _center_gram_batched(Ky)
        num = (Kx_c * Ky_c).sum(dim=(1, 2))
        den = (Kx_c.square().sum(dim=(1, 2)).sqrt() *
               Ky_c.square().sum(dim=(1, 2)).sqrt()) + torch.tensor(
                   eps, dtype=Kx_c.dtype, device=Kx_c.device
               )
        return (num / den).clamp(0.0, 1.0)
    else:
        T, L, H = X.shape
        A = X[:-1].reshape(-1, H)
        B = X[1:].reshape(-1, H)
        scores = compute_mi_rbf_cka_batched(
            torch.stack([A, B], dim=1), layer="last", sigma=sigma, eps=eps, use_double=use_double
        )
        return scores.view(T - 1, L)

# ======================================================================================
# Unified front-end
# ======================================================================================

def compute_mi_transitions(
    activations: torch.Tensor,
    *,
    layer: Optional[int | Literal["last", "all"]] = "last",
    method: Literal["linear", "rbf"] = "linear",
    sigma: float = 50.0,
    eps: float = 1e-12,
    use_double: bool = False,
    # RBF-specific:
    rbf_mode: Literal["stream", "batch"] = "stream",
    chunk_size: int = 2048,
    subsample: Optional[Union[int, float]] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Compute transition scores between consecutive tokens.
    - 'linear'  -> tiny memory, fastest.
    - 'rbf'     -> use rbf_mode='stream' to prevent OOM; optionally subsample hidden units.

    Returns:
      [T-1] for a single layer, or [T-1, L] for all layers.
    """
    if method == "linear":
        return compute_mi_linear_cka(
            activations, layer=layer, eps=eps, use_double=use_double, subsample=subsample
        )
    if method == "rbf":
        if rbf_mode == "stream":
            return compute_mi_rbf_cka_streaming(
                activations, layer=layer, sigma=sigma, eps=eps, use_double=use_double,
                chunk_size=chunk_size, subsample=subsample, show_progress=show_progress
            )
        elif rbf_mode == "batch":
            return compute_mi_rbf_cka_batched(
                activations, layer=layer, sigma=sigma, eps=eps, use_double=use_double
            )
        else:
            raise ValueError("rbf_mode must be 'stream' or 'batch'")
    raise ValueError("method must be 'linear' or 'rbf'")

def compute_mi_batch(
    x: torch.Tensor,  # [B, H]
    y: torch.Tensor,  # [B, H]
    *,
    sigma: float = 50.0,
    eps: float = 1e-12,
    use_double: bool = False,
    chunk_size: int = 2048,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    MI-like score (streaming RBF CKA) across a batch of vector pairs.

    Args:
        x, y: [B, H] tensors, same device and same shape.
        sigma, eps, use_double, chunk_size: Passed to the streaming RBF CKA.
        show_progress: If True, wraps batch loop in tqdm.

    Returns:
        scores: [B] tensor with one scalar score per row in the batch.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} vs {y.device}")
    if x.ndim != 2:
        raise ValueError(f"x and y must be 2D [B, H], got ndim={x.ndim}")

    B, H = x.shape
    scores = []
    it = range(B)
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
            it = _tqdm(it, desc="MI batch")
        except Exception:
            pass

    for i in it:
        scores.append(
            _rbf_cka_pair_streaming(
                x[i], y[i],
                sigma=sigma, eps=eps, use_double=use_double, chunk_size=chunk_size
            )
        )
    return torch.stack(scores, dim=0)  # [B]

__all__ = [
    "compute_mi_batch",
    "compute_mi_transitions",
    "compute_mi_linear_cka",
    "compute_mi_rbf_cka_streaming",
    "compute_mi_rbf_cka_batched",
]

