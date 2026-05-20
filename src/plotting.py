"""Plotting helpers for task 5 notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_current_figure(path: str | Path, dpi: int = 180) -> None:
    """Save the current matplotlib figure with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_timeseries(time: np.ndarray, data: np.ndarray, title: str) -> None:
    """Plot a one-dimensional time series."""
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, lw=0.8)
    plt.xlabel("Time")
    plt.ylabel("Strain / normalized amplitude")
    plt.title(title)
    plt.grid(alpha=0.25)


def plot_pcolormesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str = "Time",
    ylabel: str = "Frequency",
    cmap: str = "viridis",
) -> None:
    """Plot a robustly scaled pcolormesh figure."""
    values = np.asarray(z)
    finite = values[np.isfinite(values)]
    vmin, vmax = np.percentile(finite, [5, 99]) if finite.size else (None, None)

    plt.figure(figsize=(10, 5))
    mesh = plt.pcolormesh(x, y, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(mesh, label="log-scaled energy")


def plot_frequency_diagnostic(freq: np.ndarray, psd: np.ndarray, title: str) -> None:
    """Plot a log-log frequency diagnostic curve."""
    plt.figure(figsize=(9, 4))
    plt.loglog(freq, psd, lw=0.9)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title(title)
    plt.grid(alpha=0.25, which="both")
