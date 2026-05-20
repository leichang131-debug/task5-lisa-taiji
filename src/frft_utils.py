"""Fractional Fourier transform helpers based on torch-frft."""

from __future__ import annotations

import numpy as np
import torch
from torch_frft.frft_module import frft


def frft_numpy(data: np.ndarray, alpha: float) -> np.ndarray:
    """Run torch-frft on a one-dimensional NumPy array and return NumPy output."""
    tensor = torch.as_tensor(np.asarray(data), dtype=torch.complex64)
    transformed = frft(tensor, float(alpha))
    return transformed.detach().cpu().numpy()


def scan_frft_alpha(data: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Compute FRFT energy for a grid of fractional orders."""
    rows = []
    for alpha in alphas:
        transformed = frft_numpy(data, float(alpha))
        rows.append(np.abs(transformed) ** 2)
    return np.asarray(rows)
