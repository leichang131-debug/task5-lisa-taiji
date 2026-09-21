"""Run stage-1 time and GPU waveform equivalence checks on real TDC inputs."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(
    os.environ.get("TASK5_STAGE1_CACHE_DIR", REPO_ROOT / ".cache" / "stage1_tc")
).resolve()
for child in ("cupy", "cuda", "matplotlib", "numba", "torch", "tmp"):
    (CACHE_ROOT / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUPY_CACHE_DIR", str(CACHE_ROOT / "cupy"))
os.environ.setdefault("CUDA_CACHE_PATH", str(CACHE_ROOT / "cuda"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("TMPDIR", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("TMP", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRIANGLE_BBH_DIR = REPO_ROOT / "external" / "Triangle-BBH"
TRIANGLE_SIM_DIR = REPO_ROOT / "external" / "Triangle-Simulator"
if not TRIANGLE_BBH_DIR.exists():
    TRIANGLE_BBH_DIR = Path("/mnt/e/TDCEnv/Repos/Triangle-BBH")
if not TRIANGLE_SIM_DIR.exists():
    TRIANGLE_SIM_DIR = Path("/mnt/e/TDCEnv/Repos/Triangle-Simulator")
for path in (TRIANGLE_BBH_DIR, TRIANGLE_SIM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.time_conventions import (
    constellation_to_ssb_day,
    get_catalog_tc_day,
    get_window_bounds,
    ssb_to_constellation_day,
    validate_window_bounds,
)
from Triangle.Constants import DAY, TWOPI
from Triangle.Data import read_dict_from_h5
from Triangle.FFTTools import FFT_window, PSD_window
from Triangle.Orbit import Orbit
from Triangle.TDI import AETfromXYZ
from Triangle_BBH.Fisher import Likelihood
from Triangle_BBH.Response import BBHxFDTDIResponseGenerator
from Triangle_BBH.Utils import ParamDict2ParamArr, get_reflected_parameter_dict
from Triangle_BBH.Waveform import BBHxWaveformGenerator


FMIN = 0.5e-4
FMAX = 1e-2
DATA_PATH = REPO_ROOT / "data" / "tdc" / "0_2_MBHB_TDIXYZ.h5"
PARAMETER_PATH = REPO_ROOT / "data" / "tdc" / "0_2_MBHB_parameters.h5"
ORBIT_PATH = TRIANGLE_SIM_DIR / "OrbitData" / "MicroSateOrbitEclipticTCB"
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "task5_subtask2_remediation"
    / "audit"
    / "stage1_tc_definition"
)
CONVERSION_DIR = OUTPUT_ROOT / "conversions"
WAVEFORM_DIR = OUTPUT_ROOT / "waveform_checks"
FIGURE_DIR = OUTPUT_ROOT / "figures"
MANIFEST_PATH = OUTPUT_ROOT / "manifests" / "time_convention_manifest.json"

ROUNDTRIP_TOLERANCE_SECONDS = 1e-3
WAVEFORM_RELATIVE_L2_TOLERANCE = 1e-6
WAVEFORM_MISMATCH_TOLERANCE = 1e-6


def to_float(value) -> float:
    array = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
    return float(np.ravel(array)[0])


def build_window(read_dict: dict, catalog_tc_day: float, mode: str) -> dict:
    full_time = np.asarray(read_dict["time"])
    a2_td, e2_td, _ = AETfromXYZ(
        read_dict["XYZ"]["X2"], read_dict["XYZ"]["Y2"], read_dict["XYZ"]["Z2"]
    )
    full_channels_td = -np.array([a2_td, e2_td])
    start_day, end_day = get_window_bounds(catalog_tc_day, mode)
    validate_window_bounds(catalog_tc_day, mode, start_day, end_day)
    mask = (full_time / DAY >= start_day) & (full_time / DAY <= end_day)
    data_time = full_time[mask]
    data_channels_td = full_channels_td[:, mask]
    dt = float(data_time[1] - data_time[0])
    tobs = len(data_time) * dt

    spectra = []
    for channel in data_channels_td:
        frequency, spectrum = FFT_window(
            data_array=channel,
            fsample=1.0 / dt,
            window_type="tukey",
            window_args_dict={"alpha": 1000.0 / tobs},
        )
        spectra.append(spectrum)
    data_fd = np.asarray(spectra) * np.exp(-TWOPI * 1.0j * frequency * data_time[0])

    psd_mask = full_time < data_time[0]
    psd_frequency, a2_psd = PSD_window(
        data_array=a2_td[psd_mask], fsample=1.0 / dt, window_type="hann", nbin=20
    )
    _, e2_psd = PSD_window(
        data_array=e2_td[psd_mask], fsample=1.0 / dt, window_type="hann", nbin=20
    )
    psd = np.asarray(
        [
            CubicSpline(psd_frequency, a2_psd, extrapolate=True)(frequency),
            CubicSpline(psd_frequency, e2_psd, extrapolate=True)(frequency),
        ]
    )
    keep = (frequency >= FMIN) & (frequency <= FMAX)
    frequency = frequency[keep]
    data_fd = data_fd[:, keep]
    psd = psd[:, keep]
    covariance = np.asarray(
        [[psd[0], np.zeros_like(frequency)], [np.zeros_like(frequency), psd[1]]]
    ) / 4.0 * tobs
    inverse_covariance = np.linalg.inv(np.transpose(covariance, (2, 0, 1)))
    return {
        "mode": mode,
        "start_day": start_day,
        "end_day": end_day,
        "data_time": data_time,
        "frequency": frequency,
        "data_fd": data_fd,
        "inverse_covariance": inverse_covariance,
    }


def delay_function(response, base_parameters: dict):
    def delay_seconds(tc_constellation_day: float) -> float:
        parameters = copy.deepcopy(base_parameters)
        parameters["coalescence_time"] = float(tc_constellation_day)
        gpu_parameters = {
            name: cp.atleast_1d(value) for name, value in parameters.items()
        }
        wave_vector = response.WaveVector(gpu_parameters)
        delay = response.SSBToConstellationDelay(wave_vector, gpu_parameters)
        return to_float(delay)

    return delay_seconds


def time_case(
    response,
    label: str,
    parameters: dict,
    tc_ssb_day: float,
) -> dict:
    delay_seconds = delay_function(response, parameters)
    forward = ssb_to_constellation_day(tc_ssb_day, delay_seconds)
    backward = constellation_to_ssb_day(forward.output_day, delay_seconds)
    reverse = ssb_to_constellation_day(backward.output_day, delay_seconds)
    return {
        "label": label,
        "longitude_rad": float(parameters["longitude"]),
        "latitude_rad": float(parameters["latitude"]),
        "tc_ssb_input_day": float(tc_ssb_day),
        "tc_constellation_day": forward.output_day,
        "delay_seconds": forward.delay_seconds,
        "forward_iterations": forward.iterations,
        "forward_converged": forward.converged,
        "ssb_roundtrip_day": backward.output_day,
        "ssb_roundtrip_error_seconds": (backward.output_day - tc_ssb_day) * DAY,
        "constellation_roundtrip_error_seconds": (
            reverse.output_day - forward.output_day
        )
        * DAY,
    }


def noise_inner(a: np.ndarray, b: np.ndarray, inverse_covariance: np.ndarray) -> complex:
    return np.einsum("if,fij,jf->", np.conjugate(a), inverse_covariance, b)


def waveform_case(
    response,
    window: dict,
    parameters: dict,
    tc_ssb_day: float,
    interpolation: bool,
) -> dict:
    delay_seconds = delay_function(response, parameters)
    conversion = ssb_to_constellation_day(tc_ssb_day, delay_seconds)
    if not conversion.converged:
        raise RuntimeError("SSB-to-constellation fixed-point iteration did not converge")

    parameters_ssb = copy.deepcopy(parameters)
    parameters_ssb["coalescence_time"] = float(tc_ssb_day)
    parameters_constellation = copy.deepcopy(parameters)
    parameters_constellation["coalescence_time"] = conversion.output_day
    frequency_gpu = cp.asarray(window["frequency"])
    common = {
        "modes": [(2, 2)],
        "tmin": window["data_time"][0] / DAY,
        "tmax": window["data_time"][-1] / DAY,
        "TDIGeneration": "2nd",
        "optimal_combination": True,
        "drop_T": True,
        "interpolation": interpolation,
    }
    start = time.perf_counter()
    waveform_ssb_gpu = response.Response(
        parameters=parameters_ssb,
        freqs=frequency_gpu,
        tc_at_constellation=False,
        **common,
    )
    cp.cuda.Stream.null.synchronize()
    ssb_seconds = time.perf_counter() - start
    start = time.perf_counter()
    waveform_constellation_gpu = response.Response(
        parameters=parameters_constellation,
        freqs=frequency_gpu,
        tc_at_constellation=True,
        **common,
    )
    cp.cuda.Stream.null.synchronize()
    constellation_seconds = time.perf_counter() - start

    waveform_ssb = cp.asnumpy(waveform_ssb_gpu)
    waveform_constellation = cp.asnumpy(waveform_constellation_gpu)
    difference = waveform_constellation - waveform_ssb
    reference_norm = np.linalg.norm(waveform_ssb)
    relative_l2 = float(np.linalg.norm(difference) / reference_norm)
    scale = float(np.max(np.abs(waveform_ssb)))
    max_relative_absolute = float(np.max(np.abs(difference)) / scale)
    valid = np.abs(waveform_ssb) > max(scale * 1e-12, 1e-30)
    maximum_phase_residual = float(
        np.max(np.abs(np.angle(waveform_constellation[valid] / waveform_ssb[valid])))
    )

    inverse_covariance = window["inverse_covariance"]
    aa = float(np.real(noise_inner(waveform_ssb, waveform_ssb, inverse_covariance)))
    bb = float(
        np.real(
            noise_inner(
                waveform_constellation, waveform_constellation, inverse_covariance
            )
        )
    )
    ab = float(
        np.real(noise_inner(waveform_ssb, waveform_constellation, inverse_covariance))
    )
    overlap = ab / np.sqrt(aa * bb)
    mismatch = float(1.0 - overlap)

    data_gpu = cp.asarray(window["data_fd"])
    inverse_covariance_gpu = cp.asarray(inverse_covariance)
    likelihood_ssb = Likelihood(
        response_generator=response,
        frequency=frequency_gpu,
        data=data_gpu,
        invserse_covariance_matrix=inverse_covariance_gpu,
        response_parameters={**common, "tc_at_constellation": False},
        use_gpu=True,
    )
    likelihood_constellation = Likelihood(
        response_generator=response,
        frequency=frequency_gpu,
        data=data_gpu,
        invserse_covariance_matrix=inverse_covariance_gpu,
        response_parameters={**common, "tc_at_constellation": True},
        use_gpu=True,
    )
    logl_ssb = float(likelihood_ssb.full_log_like(ParamDict2ParamArr(parameters_ssb)))
    logl_constellation = float(
        likelihood_constellation.full_log_like(
            ParamDict2ParamArr(parameters_constellation)
        )
    )
    return {
        "window": window["mode"],
        "interpolation": bool(interpolation),
        "frequency_bins": int(len(window["frequency"])),
        "tc_ssb_day": float(tc_ssb_day),
        "tc_constellation_day": conversion.output_day,
        "delay_seconds": conversion.delay_seconds,
        "relative_l2": relative_l2,
        "max_relative_absolute": max_relative_absolute,
        "maximum_phase_residual_rad": maximum_phase_residual,
        "noise_weighted_overlap": float(overlap),
        "noise_weighted_mismatch": mismatch,
        "full_log_likelihood_ssb": logl_ssb,
        "full_log_likelihood_constellation": logl_constellation,
        "full_log_likelihood_delta": logl_constellation - logl_ssb,
        "ssb_waveform_seconds": ssb_seconds,
        "constellation_waveform_seconds": constellation_seconds,
        "passed": bool(
            relative_l2 < WAVEFORM_RELATIVE_L2_TOLERANCE
            and abs(mismatch) < WAVEFORM_MISMATCH_TOLERANCE
        ),
    }


def save_json(path: Path, payload: object) -> None:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialise {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=convert) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for directory in (CONVERSION_DIR, WAVEFORM_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA_PATH, "r") as handle:
        data = read_dict_from_h5(handle["/"])
    with h5py.File(PARAMETER_PATH, "r") as handle:
        parameters = read_dict_from_h5(handle["/"])
    catalog_tc_day = get_catalog_tc_day(parameters)

    orbit = Orbit(OrbitDir=str(ORBIT_PATH))
    reflected = get_reflected_parameter_dict(parameters, orbit)
    rng = np.random.default_rng(20260921)
    time_cases = [("catalog_direct", copy.deepcopy(parameters))]
    time_cases.append(("catalog_reflected", reflected))
    for index in range(3):
        random_parameters = copy.deepcopy(parameters)
        random_parameters["longitude"] = float(rng.uniform(0.0, TWOPI))
        random_parameters["latitude"] = float(
            np.arcsin(rng.uniform(-1.0, 1.0))
        )
        random_parameters["psi"] = float(rng.uniform(0.0, np.pi))
        time_cases.append((f"random_sky_{index + 1}", random_parameters))

    waveform_generator = BBHxWaveformGenerator(mode="primary", use_gpu=True)
    response = BBHxFDTDIResponseGenerator(
        orbit_class=orbit, waveform_generator=waveform_generator, use_gpu=True
    )
    roundtrip_rows = [
        time_case(response, label, case_parameters, catalog_tc_day)
        for label, case_parameters in time_cases
    ]
    roundtrip_frame = pd.DataFrame(roundtrip_rows)
    roundtrip_frame["passed"] = (
        roundtrip_frame["forward_converged"]
        & (
            roundtrip_frame["ssb_roundtrip_error_seconds"].abs()
            < ROUNDTRIP_TOLERANCE_SECONDS
        )
        & (
            roundtrip_frame["constellation_roundtrip_error_seconds"].abs()
            < ROUNDTRIP_TOLERANCE_SECONDS
        )
    )
    roundtrip_frame.to_csv(CONVERSION_DIR / "time_roundtrip_results.csv", index=False)
    save_json(CONVERSION_DIR / "time_roundtrip_results.json", roundtrip_frame.to_dict("records"))

    windows = {
        mode: build_window(data, catalog_tc_day, mode)
        for mode in ("official_baseline", "task_five_day")
    }
    waveform_rows = []
    for window in windows.values():
        for interpolation in (False, True):
            waveform_rows.append(
                waveform_case(
                    response,
                    window,
                    parameters,
                    catalog_tc_day,
                    interpolation,
                )
            )
    waveform_frame = pd.DataFrame(waveform_rows)
    waveform_frame.to_csv(WAVEFORM_DIR / "gpu_waveform_equivalence.csv", index=False)
    save_json(WAVEFORM_DIR / "gpu_waveform_equivalence.json", waveform_frame.to_dict("records"))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [
        f"{row.window}\ninterp={row.interpolation}"
        for row in waveform_frame.itertuples()
    ]
    plot_floor = 1e-18
    relative_l2_plot = waveform_frame["relative_l2"].clip(lower=plot_floor)
    mismatch_plot = waveform_frame["noise_weighted_mismatch"].abs().clip(
        lower=plot_floor
    )
    axes[0].bar(labels, relative_l2_plot)
    axes[0].axhline(WAVEFORM_RELATIVE_L2_TOLERANCE, color="tab:red", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative L2 residual")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(plot_floor / 2.0, WAVEFORM_RELATIVE_L2_TOLERANCE * 10.0)
    axes[1].bar(labels, mismatch_plot)
    axes[1].axhline(WAVEFORM_MISMATCH_TOLERANCE, color="tab:red", linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("absolute noise-weighted mismatch")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(plot_floor / 2.0, WAVEFORM_MISMATCH_TOLERANCE * 10.0)
    for axis, values in (
        (axes[0], waveform_frame["relative_l2"]),
        (axes[1], waveform_frame["noise_weighted_mismatch"].abs()),
    ):
        for index, value in enumerate(values):
            if value == 0.0:
                axis.text(
                    index,
                    plot_floor * 1.4,
                    "exact zero",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "stage1_gpu_waveform_equivalence.png", dpi=180)
    plt.close(figure)

    roundtrip_passed = bool(roundtrip_frame["passed"].all())
    waveform_passed = bool(waveform_frame["passed"].all())
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    report = {
        "status": "passed" if roundtrip_passed and waveform_passed else "failed",
        "cache_root": str(CACHE_ROOT),
        "cuda_device": str(device_name),
        "catalog_tc_used_as_test_ssb_day": catalog_tc_day,
        "catalog_reference_frame_asserted": False,
        "time_roundtrip_passed": roundtrip_passed,
        "gpu_waveform_equivalence_passed": waveform_passed,
        "roundtrip_tolerance_seconds": ROUNDTRIP_TOLERANCE_SECONDS,
        "waveform_relative_l2_tolerance": WAVEFORM_RELATIVE_L2_TOLERANCE,
        "waveform_mismatch_tolerance": WAVEFORM_MISMATCH_TOLERANCE,
        "maximum_roundtrip_error_seconds": float(
            max(
                roundtrip_frame["ssb_roundtrip_error_seconds"].abs().max(),
                roundtrip_frame["constellation_roundtrip_error_seconds"].abs().max(),
            )
        ),
        "maximum_waveform_relative_l2": float(waveform_frame["relative_l2"].max()),
        "maximum_absolute_mismatch": float(
            waveform_frame["noise_weighted_mismatch"].abs().max()
        ),
    }
    save_json(OUTPUT_ROOT / "stage1_closed_loop_report.json", report)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = "closed_loop_tests_passed" if report["status"] == "passed" else "closed_loop_tests_failed"
    manifest["closed_loop_tests"] = report
    manifest["accelerated_gpu_route"]["time_conversion_contract"] = (
        "SSB tc with tc_at_constellation=False is numerically identical to "
        "SSB-to-constellation converted tc with tc_at_constellation=True"
    )
    manifest["accelerated_gpu_route"]["final_time_frame_decision"] = (
        "conversion_contract_verified_catalog_source_frame_unresolved"
        if report["status"] == "passed"
        else "closed_loop_verification_failed"
    )
    save_json(MANIFEST_PATH, manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
