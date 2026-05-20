"""Data loading helpers for LDC/Taiji task 5 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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


def read_tdi_dataset(path: str | Path, dataset_name: str) -> dict[str, np.ndarray]:
    """Read an LDC compound TDI dataset with fields t, X, Y, and Z."""
    data = read_dataset(path, dataset_name).reshape(-1)
    return {field: np.asarray(data[field]) for field in data.dtype.names}


def read_dataset_attrs(path: str | Path, dataset_name: str) -> dict[str, Any]:
    """Read attributes from one HDF5 dataset."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return dict(handle[dataset_name].attrs)


def read_sky_catalog(path: str | Path) -> dict[str, Any]:
    """Read the scalar sky/catalog record as a plain dictionary."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        record = handle["sky/cat"][()]
    return {name: record[name].item() for name in record.dtype.names}


def clean_timeseries(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Replace NaNs with zero, remove mean, and optionally standardize."""
    cleaned = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
    cleaned = cleaned - np.mean(cleaned)
    if normalize:
        std = np.std(cleaned)
        if std > 0:
            cleaned = cleaned / std
    return cleaned


def tdi_xyz_to_aet(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    """Construct commonly used orthogonal A/E/T-like channels from X/Y/Z."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    return {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }


def crop_by_time(
    time: np.ndarray,
    data: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a time series by absolute start/end times."""
    mask = (time >= start) & (time <= end)
    return np.asarray(time[mask]), np.asarray(data[mask])


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


def summarize_nan_counts(tdi: dict[str, np.ndarray], channels: Iterable[str] = ("X", "Y", "Z")) -> dict[str, int]:
    """Count NaNs in selected TDI channels."""
    return {channel: int(np.isnan(tdi[channel]).sum()) for channel in channels}
