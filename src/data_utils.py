"""Data loading helpers for LDC/Taiji task 5 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def print_hdf5_tree(path: str | Path) -> None:
    """Print groups, datasets, shapes, dtypes, and attributes in an HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}- {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}+ {name}/")
            for key, value in obj.attrs.items():
                print(f"{indent}  @{key}: {value}")

        handle.visititems(visitor)


def read_dataset(path: str | Path, dataset_name: str) -> np.ndarray:
    """Read one HDF5 dataset into a NumPy array."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset_name])


def clean_timeseries(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Replace NaNs with zero, remove mean, and optionally standardize."""
    cleaned = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
    cleaned = cleaned - np.mean(cleaned)
    if normalize:
        std = np.std(cleaned)
        if std > 0:
            cleaned = cleaned / std
    return cleaned


def crop_to_wdm_shape(data: np.ndarray, n_freq: int) -> tuple[np.ndarray, int]:
    """Crop a one-dimensional array so that len(data) = n_freq * n_time."""
    if n_freq <= 0:
        raise ValueError("n_freq must be positive")
    n_time = len(data) // n_freq
    if n_time % 2 == 1:
        n_time -= 1
    if n_time <= 0:
        raise ValueError("data is too short for the requested n_freq")
    usable = n_freq * n_time
    return np.asarray(data[:usable]), n_time
