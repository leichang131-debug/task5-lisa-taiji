"""GPU preflight for Task 5 subtask 2.

This script runs only the short official-GPU-path checks:
load real TDC data, build the official Example-5-style 5-day baseline window,
instantiate BBHx GPU waveform/response objects, evaluate one injected waveform,
and evaluate the heterodyned likelihood at the injected parameters.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cupy as xp
import h5py
import numpy as np
from scipy.interpolate import CubicSpline

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIANGLE_BBH_DIR = REPO_ROOT / "external" / "Triangle-BBH"
TRIANGLE_SIM_DIR = REPO_ROOT / "external" / "Triangle-Simulator"
if not TRIANGLE_BBH_DIR.exists():
    TRIANGLE_BBH_DIR = Path("/mnt/e/TDCEnv/Repos/Triangle-BBH")
if not TRIANGLE_SIM_DIR.exists():
    TRIANGLE_SIM_DIR = Path("/mnt/e/TDCEnv/Repos/Triangle-Simulator")
for path in (TRIANGLE_BBH_DIR, TRIANGLE_SIM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Triangle.Constants import DAY, TWOPI
from Triangle.Data import read_dict_from_h5
from Triangle.FFTTools import FFT_window, PSD_window
from Triangle.Orbit import Orbit
from Triangle.TDI import AETfromXYZ
from Triangle_BBH.Response import BBHxFDTDIResponseGenerator
from Triangle_BBH.Utils import ParamDict2ParamArr
from Triangle_BBH.Waveform import BBHxWaveformGenerator
from Triangle_BBH.Fisher import Likelihood

FMIN = 0.5e-4
FMAX = 1e-2
DATA_DIR = REPO_ROOT / "data" / "tdc" / "0_2_MBHB_TDIXYZ.h5"
PARAM_DIR = REPO_ROOT / "data" / "tdc" / "0_2_MBHB_parameters.h5"
ORBIT_DIR = TRIANGLE_SIM_DIR / "OrbitData" / "MicroSateOrbitEclipticTCB"
RESULT_DIR = REPO_ROOT / "results" / "task5_subtask2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_window(read_dict: dict, injected_parameters: dict) -> dict:
    full_time = np.asarray(read_dict["time"])
    a2_td, e2_td, _ = AETfromXYZ(read_dict["XYZ"]["X2"], read_dict["XYZ"]["Y2"], read_dict["XYZ"]["Z2"])
    full_channels_td = -np.array([a2_td, e2_td])
    tc_day = float(injected_parameters["coalescence_time"])
    mask = np.abs(full_time / DAY - tc_day) < 2.5
    data_time = full_time[mask]
    data_channels_td = full_channels_td[:, mask]
    dt = float(data_time[1] - data_time[0])
    tobs = len(data_time) * dt

    data_channels_fd = []
    for channel in data_channels_td:
        ff, xf = FFT_window(
            data_array=channel,
            fsample=1.0 / dt,
            window_type="tukey",
            window_args_dict={"alpha": 1000.0 / tobs},
        )
        data_channels_fd.append(xf)
    data_channels_fd = np.array(data_channels_fd) * np.exp(-TWOPI * 1.0j * ff * data_time[0])
    data_frequency = ff

    psd_mask = full_time < data_time[0]
    ff_psd, a2_psd = PSD_window(data_array=a2_td[psd_mask], fsample=1.0 / dt, window_type="hann", nbin=20)
    _, e2_psd = PSD_window(data_array=e2_td[psd_mask], fsample=1.0 / dt, window_type="hann", nbin=20)
    psd_channels = np.array(
        [
            CubicSpline(ff_psd, a2_psd, extrapolate=True)(data_frequency),
            CubicSpline(ff_psd, e2_psd, extrapolate=True)(data_frequency),
        ]
    )

    freq_idx = np.where((data_frequency >= FMIN) & (data_frequency <= FMAX))[0]
    data_frequency = data_frequency[freq_idx]
    data_channels_fd = data_channels_fd[:, freq_idx]
    psd_channels = psd_channels[:, freq_idx]

    covmat = np.array(
        [
            [psd_channels[0], np.zeros_like(data_frequency)],
            [np.zeros_like(data_frequency), psd_channels[1]],
        ]
    ) / 4.0 * tobs
    inv_covmat = np.linalg.inv(np.transpose(covmat, (2, 0, 1)))
    return {
        "data_time": data_time,
        "data_frequency": data_frequency,
        "data_channels_fd": data_channels_fd,
        "InvCovMat": inv_covmat,
        "Tobs": tobs,
        "tc_day": tc_day,
    }


def main() -> None:
    with h5py.File(DATA_DIR, "r") as h5file:
        read_dict = read_dict_from_h5(h5file["/"])
    with h5py.File(PARAM_DIR, "r") as h5file:
        injected_parameters = read_dict_from_h5(h5file["/"])

    window = build_window(read_dict, injected_parameters)
    orbit = Orbit(OrbitDir=str(ORBIT_DIR))
    wfg = BBHxWaveformGenerator(mode="primary", use_gpu=True)
    fdtdi = BBHxFDTDIResponseGenerator(orbit_class=orbit, waveform_generator=wfg, use_gpu=True)
    frequency = xp.asarray(window["data_frequency"])
    data = xp.asarray(window["data_channels_fd"])
    inv_cov = xp.asarray(window["InvCovMat"])
    response_kwargs_interp = {
        "modes": [(2, 2)],
        "tmin": window["data_time"][0] / DAY,
        "tmax": window["data_time"][-1] / DAY,
        "tc_at_constellation": True,
        "TDIGeneration": "2nd",
        "optimal_combination": True,
        "drop_T": True,
        "interpolation": True,
    }
    response_kwargs_direct = dict(response_kwargs_interp)
    response_kwargs_direct["interpolation"] = False

    waveform = fdtdi.Response(parameters=injected_parameters, freqs=frequency, **response_kwargs_interp)
    like = Likelihood(
        response_generator=fdtdi,
        frequency=frequency,
        data=data,
        invserse_covariance_matrix=inv_cov,
        response_parameters=response_kwargs_direct,
        use_gpu=True,
    )
    injected_array = ParamDict2ParamArr(injected_parameters)
    like.prepare_het_log_like(base_parameters=injected_array)
    log_likelihood = float(like.het_log_like(parameter_array=injected_array))

    residual = data - waveform
    report = {
        "cuda_device_count": int(xp.cuda.runtime.getDeviceCount()),
        "frequency_bins": int(frequency.shape[0]),
        "data_shape": [int(i) for i in data.shape],
        "waveform_shape": [int(i) for i in waveform.shape],
        "data_norm": float(xp.linalg.norm(data).get()),
        "waveform_norm": float(xp.linalg.norm(waveform).get()),
        "residual_norm": float(xp.linalg.norm(residual).get()),
        "heterodyned_log_likelihood_at_injection": log_likelihood,
        "cuda_home": os.environ.get("CUDAHOME"),
        "cuda_path": os.environ.get("CUDA_PATH"),
    }
    (RESULT_DIR / "baseline_example4_gpu_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
