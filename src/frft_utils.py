"""Fractional Fourier transform helpers.

The one-dimensional FRFT implementation follows the fast chirp-convolution
algorithm used by nanaln/python_frft, with small updates for modern NumPy
(`np.complex128` instead of the removed `np.complex` alias).
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def sincinterp(x: np.ndarray) -> np.ndarray:
    """Sinc-interpolate a one-dimensional signal onto a half-step grid."""
    n = len(x)
    y = np.zeros(2 * n - 1, dtype=x.dtype)
    y[: 2 * n : 2] = x
    kernel = np.sinc(np.arange(-(2 * n - 3), 2 * n - 2).T / 2)
    interpolated = scipy.signal.fftconvolve(y[: 2 * n], kernel)
    return interpolated[2 * n - 3 : -2 * n + 3]


def frft_numpy(data: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the one-dimensional fast fractional Fourier transform."""
    f = np.asarray(data, dtype=np.complex128).copy()
    n = len(f)
    result = np.zeros_like(f, dtype=np.complex128)
    shift = np.fmod(np.arange(n) + np.fix(n / 2), n).astype(int)
    sqrt_n = np.sqrt(n)
    alpha = np.remainder(alpha, 4.0)

    if alpha == 0.0:
        return f
    if alpha == 2.0:
        return np.flipud(f)
    if alpha == 1.0:
        result[shift] = np.fft.fft(f[shift]) / sqrt_n
        return result
    if alpha == 3.0:
        result[shift] = np.fft.ifft(f[shift]) * sqrt_n
        return result

    if alpha > 2.0:
        alpha -= 2.0
        f = np.flipud(f)
    if alpha > 1.5:
        alpha -= 1.0
        f[shift] = np.fft.fft(f[shift]) / sqrt_n
    if alpha < 0.5:
        alpha += 1.0
        f[shift] = np.fft.ifft(f[shift]) * sqrt_n

    angle = alpha * np.pi / 2
    tan_half = np.tan(angle / 2)
    sin_angle = np.sin(angle)

    f = np.hstack((np.zeros(n - 1), sincinterp(f), np.zeros(n - 1))).T
    chirp = np.exp(-1j * np.pi / n * tan_half / 4 * np.arange(-2 * n + 2, 2 * n - 1).T**2)
    f = chirp * f

    c = np.pi / n / sin_angle / 4
    result = scipy.signal.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * n - 4), 4 * n - 3).T**2),
        f,
    )
    result = result[4 * n - 4 : 8 * n - 7] * np.sqrt(c / np.pi)
    result = chirp * result
    result = np.exp(-1j * (1 - alpha) * np.pi / 4) * result[n - 1 : -n + 1 : 2]
    return result


def scan_frft_alpha(data: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Compute FRFT energy for a grid of fractional orders."""
    rows = []
    for alpha in alphas:
        transformed = frft_numpy(data, float(alpha))
        rows.append(np.abs(transformed) ** 2)
    return np.asarray(rows)
