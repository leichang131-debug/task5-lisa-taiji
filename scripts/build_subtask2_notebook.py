"""Build the subtask 2 notebook from structured cells.

This script keeps the Triangle-BBH Example 4 migration reproducible.
It writes notebooks/02_taiji_mbhb_parameter_estimation.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "02_taiji_mbhb_parameter_estimation.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# Subtask 2: Taiji MBHB Parameter Estimation

This notebook implements UCAS 2026 Task 5 subtask 2 by migrating the official
Triangle-BBH Example 4 workflow into this repository and then adding the required
asymmetric 5-day window experiment.

Baseline source: `external/Triangle-BBH/Examples/4_TDC_Verification_MBHB_Search_and_Estimation(CPU).ipynb`.

Implemented requirements:

1. Reproduce official Example 4 as the baseline.
2. Record and explain TDI loading, FFT, MBHB waveform generation, F-statistics search, Fisher analysis, heterodyned likelihood, and Bayesian sampling.
3. Change the data window from the official symmetric 5-day window, `tc - 2.5 days` to `tc + 2.5 days`, to the task-required asymmetric window, `tc - 4 days` to `tc + 1 day`.
4. Rebuild all data-dependent objects for the modified window and rerun the same inference chain.
5. Save figures and summaries under `figures/task5_subtask2/` and `results/task5_subtask2/`.

Heavy search and sampling cells are controlled by runtime switches so the notebook can be opened safely.
"""
    ),
    md(
        """
## 0. Execution Checklist

- [ ] Configure `0_2_MBHB_TDIXYZ.h5` and `0_2_MBHB_parameters.h5`.
- [ ] Load TDC II TDI XYZ data and injected parameters.
- [ ] Convert XYZ to A/E/T and keep A/E channels with the official sign convention.
- [ ] Build official baseline window: `tc - 2.5 days` to `tc + 2.5 days`.
- [ ] Reproduce Example 4 FFT, PSD, frequency cut, covariance, model setup, F-statistics search, Fisher analysis, likelihood, and sampler.
- [ ] Build task-required window: `tc - 4 days` to `tc + 1 day`.
- [ ] Rebuild time-domain data, FFT, PSD, covariance, waveform response, likelihood, and sampler for the modified window.
- [ ] Compare posterior medians and 90% credible intervals.
- [ ] Update README with final figures and quantitative conclusions.
"""
    ),
    md("## 1. Environment and Reproducibility"),
    code(
        r"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent

TRIANGLE_BBH_DIR = REPO_ROOT / "external" / "Triangle-BBH"
TRIANGLE_SIM_DIR = REPO_ROOT / "external" / "Triangle-Simulator"
FIGURE_DIR = REPO_ROOT / "figures" / "task5_subtask2"
RESULT_DIR = REPO_ROOT / "results" / "task5_subtask2"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def git_commit(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Repo root:", REPO_ROOT)
print("Triangle-BBH commit:", git_commit(TRIANGLE_BBH_DIR))
print("Triangle-Simulator commit:", git_commit(TRIANGLE_SIM_DIR))
"""
    ),
    md(
        """
## 2. Runtime Switches and Paths

Official data links from Example 4:

- TDI data: https://zenodo.org/records/15469724/files/0_2_MBHB_TDIXYZ.h5?download=1
- parameters: https://zenodo.org/records/15532090/files/0_2_MBHB_parameters.h5?download=1
"""
    ),
    code(
        r"""
RUN_BASELINE_SEARCH = False
RUN_BASELINE_SAMPLER = False
RUN_FIVE_DAY_SEARCH = False
RUN_FIVE_DAY_SAMPLER = False
USE_SMOKE_TEST_SAMPLER = True

FMIN = 0.5e-4
FMAX = 1e-2
OFFICIAL_SAMPLER_SETTINGS = dict(sampler="nessai", nlive=1200, stopping=0.1)
SMOKE_TEST_SAMPLER_SETTINGS = dict(sampler="nessai", nlive=100, stopping=0.5)

CANDIDATE_TDC_ROOTS = [
    REPO_ROOT / "data" / "tdc",
    Path(r"E:\BaiduNetdiskDownload"),
    Path(r"E:\BaiduNetdiskDownload\TDCData"),
]

def find_first_existing(filename: str) -> Path | None:
    for root in CANDIDATE_TDC_ROOTS:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None

DATA_DIR = find_first_existing("0_2_MBHB_TDIXYZ.h5") or (REPO_ROOT / "data" / "tdc" / "0_2_MBHB_TDIXYZ.h5")
PARAM_DIR = find_first_existing("0_2_MBHB_parameters.h5") or (REPO_ROOT / "data" / "tdc" / "0_2_MBHB_parameters.h5")
ORBIT_DIR = TRIANGLE_SIM_DIR / "OrbitData" / "MicroSateOrbitEclipticTCB"

print("DATA_DIR:", DATA_DIR, DATA_DIR.exists())
print("PARAM_DIR:", PARAM_DIR, PARAM_DIR.exists())
print("ORBIT_DIR:", ORBIT_DIR, ORBIT_DIR.exists())
"""
    ),
    md("## 3. Imports Matching Official Example 4"),
    code(
        r"""
import bilby
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution
from tqdm import tqdm

from Triangle.Constants import *
from Triangle.Orbit import *
from Triangle.Noise import *
from Triangle.FFTTools import *
from Triangle.TDI import *
from Triangle.Data import *

from Triangle_BBH.Waveform import *
from Triangle_BBH.Response import *
from Triangle_BBH.Utils import *
from Triangle_BBH.Fisher import *

try:
    import nessai  # noqa: F401
    HAS_NESSAI = True
except Exception as exc:
    HAS_NESSAI = False
    print("NESSAI unavailable:", repr(exc))

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 220, "axes.grid": True, "grid.alpha": 0.25})
print("Imports OK")
print("bilby:", getattr(bilby, "__version__", "unknown"))
print("NESSAI available:", HAS_NESSAI)
"""
    ),
    md("## 4. Utility Functions"),
    code(
        r"""
def save_current_figure(filename: str) -> Path:
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path.relative_to(REPO_ROOT)}")
    return path

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def print_h5_tree(path: Path, max_items: int = 80) -> None:
    require_file(path, "HDF5 file")
    count = 0
    with h5py.File(path, "r") as h5:
        def visitor(name, obj):
            nonlocal count
            if count >= max_items:
                return
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET /{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"GROUP   /{name}")
            count += 1
        h5.visititems(visitor)
    if count >= max_items:
        print(f"... stopped after {max_items} items")

def save_json(obj, filename: str) -> Path:
    path = RESULT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {path.relative_to(REPO_ROOT)}")
    return path

def save_parameter_dict(param_dict: dict, filename: str) -> Path:
    clean = {k: (float(v) if np.isscalar(v) and v is not None else v) for k, v in param_dict.items()}
    return save_json(clean, filename)

def posterior_summary(samples: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    rows = []
    for p in parameters:
        if p not in samples:
            continue
        q05, q50, q95 = np.percentile(samples[p], [5, 50, 95])
        rows.append(dict(parameter=p, median=q50, ci90_low=q05, ci90_high=q95, ci90_width=q95-q05))
    return pd.DataFrame(rows)

def is_valid_hdf5(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r"):
            return True
    except OSError as exc:
        print(f"Invalid or incomplete HDF5 file: {path}")
        print("h5py error:", exc)
        return False
"""
    ),
    md("## 5. Inspect and Load TDC Data"),
    code(
        r"""
DATA_FILE_OK = is_valid_hdf5(DATA_DIR)
PARAM_FILE_OK = is_valid_hdf5(PARAM_DIR)

if DATA_FILE_OK:
    print("Data file tree:")
    print_h5_tree(DATA_DIR)
else:
    print("Data file is not present or is incomplete:", DATA_DIR)

if PARAM_FILE_OK:
    print("\nParameter file tree:")
    print_h5_tree(PARAM_DIR)
else:
    print("Parameter file is not present or is incomplete:", PARAM_DIR)
"""
    ),
    code(
        r"""
read_dict = None
injected_parameters = None

if DATA_FILE_OK and PARAM_FILE_OK:
    with h5py.File(DATA_DIR, "r") as h5file:
        read_dict = read_dict_from_h5(h5file["/"])
    with h5py.File(PARAM_DIR, "r") as h5file:
        injected_parameters = read_dict_from_h5(h5file["/"])
    print("read_dict keys:", read_dict.keys())
    print("injected_parameters keys:", injected_parameters.keys())
else:
    print("Skipping data load because one or both TDC files are missing or incomplete.")
"""
    ),
    md(
        """
## 6. Convert TDI XYZ to A/E Channels

This follows official Example 4: construct A/E/T, drop T, and apply the minus sign because the Michelson TDI-2.0 convention differs from Triangle-Simulator by a minus sign.
"""
    ),
    code(
        r"""
full_time = None
full_channels_td = None
full_A2_td = None
full_E2_td = None
channel_names = ["A2", "E2"]

if read_dict is not None:
    full_time = np.asarray(read_dict["time"])
    full_A2_td, full_E2_td, _ = AETfromXYZ(read_dict["XYZ"]["X2"], read_dict["XYZ"]["Y2"], read_dict["XYZ"]["Z2"])
    full_channels_td = -np.array([full_A2_td, full_E2_td])
    full_dt = float(full_time[1] - full_time[0])
    print("full_time shape:", full_time.shape)
    print("full_channels_td shape:", full_channels_td.shape)
    print("dt:", full_dt, "s")
else:
    print("No TDC data loaded yet.")
"""
    ),
    md(
        """
## 7. Window Builder Shared by Baseline and Modified Run

Official Example 4 uses `abs(data_time / DAY - tc) < 2.5`. The task-required run uses `tc - 4 days` to `tc + 1 day`. The function below rebuilds FFT, PSD, frequency cut, covariance, and inverse covariance for each window.
"""
    ),
    code(
        r"""
def get_tc_day(params: dict) -> float:
    return float(params["coalescence_time"])

def get_window_bounds(tc_day: float, mode: str) -> tuple[float, float]:
    if mode == "official_baseline":
        return tc_day - 2.5, tc_day + 2.5
    if mode == "task_five_day":
        return tc_day - 4.0, tc_day + 1.0
    raise ValueError(f"Unknown mode: {mode}")

def build_window_data(label: str, mode: str, psd_mode: str = "before") -> dict:
    if full_time is None or full_channels_td is None or injected_parameters is None:
        raise RuntimeError("Load TDC data and injected parameters first.")
    tc_day = get_tc_day(injected_parameters)
    start_day, end_day = get_window_bounds(tc_day, mode)
    mask = (full_time / DAY >= start_day) & (full_time / DAY <= end_day)
    if mask.sum() < 16:
        raise ValueError(f"{label}: selected window has too few samples: {mask.sum()}")
    data_time = full_time[mask]
    data_channels_td = full_channels_td[:, mask]
    dt = float(data_time[1] - data_time[0])
    Tobs = len(data_time) * dt

    data_channels_fd = []
    for i in range(len(data_channels_td)):
        ff, xf = FFT_window(data_array=data_channels_td[i], fsample=1.0/dt, window_type="tukey", window_args_dict=dict(alpha=1000.0/Tobs))
        data_channels_fd.append(xf)
    data_channels_fd = np.array(data_channels_fd) * np.exp(-TWOPI * 1.j * ff * data_time[0])
    data_frequency = ff

    if psd_mode == "before":
        psd_mask = full_time < data_time[0]
    elif psd_mode == "outside":
        psd_mask = (full_time < data_time[0]) | (full_time > data_time[-1])
    else:
        raise ValueError("psd_mode must be 'before' or 'outside'")
    if psd_mask.sum() < 16:
        raise ValueError(f"{label}: not enough silent samples for PSD: {psd_mask.sum()}")

    ff_psd, A2_PSD = PSD_window(data_array=full_A2_td[psd_mask], fsample=1.0/dt, window_type="hann", nbin=20)
    _, E2_PSD = PSD_window(data_array=full_E2_td[psd_mask], fsample=1.0/dt, window_type="hann", nbin=20)
    psd_channels = np.array([CubicSpline(ff_psd, A2_PSD, extrapolate=True)(data_frequency), CubicSpline(ff_psd, E2_PSD, extrapolate=True)(data_frequency)])

    freq_idx = np.where((data_frequency >= FMIN) & (data_frequency <= FMAX))[0]
    data_frequency = data_frequency[freq_idx]
    data_channels_fd = data_channels_fd[:, freq_idx]
    psd_channels = psd_channels[:, freq_idx]

    CovMat = np.array([[psd_channels[0], np.zeros_like(data_frequency)], [np.zeros_like(data_frequency), psd_channels[1]]]) / 4.0 * Tobs
    InvCovMat = np.linalg.inv(np.transpose(CovMat, (2, 0, 1)))
    return dict(label=label, mode=mode, tc_day=tc_day, start_day=start_day, end_day=end_day, data_time=data_time, data_channels_td=data_channels_td, dt=dt, Tobs=Tobs, data_frequency=data_frequency, data_channels_fd=data_channels_fd, psd_channels=psd_channels, CovMat=CovMat, InvCovMat=InvCovMat, psd_mode=psd_mode, psd_samples=int(psd_mask.sum()))

def print_window_summary(window: dict) -> None:
    print(f"[{window['label']}]")
    print("mode:", window["mode"])
    print("tc_day:", window["tc_day"])
    print("start_day/end_day:", window["start_day"], window["end_day"])
    print("duration_days:", (window["data_time"][-1] - window["data_time"][0]) / DAY)
    print("dt:", window["dt"])
    print("Tobs:", window["Tobs"])
    print("df median:", np.median(np.diff(window["data_frequency"])))
    print("expected 1/Tobs:", 1.0 / window["Tobs"])
    print("data_td:", window["data_channels_td"].shape)
    print("data_fd:", window["data_channels_fd"].shape)
    print("InvCovMat:", window["InvCovMat"].shape)
    print("PSD mode/samples:", window["psd_mode"], window["psd_samples"])
"""
    ),
    md("## 8. Build and Plot Official Baseline Window"),
    code(
        r"""
baseline_window = None
if read_dict is not None:
    baseline_window = build_window_data("baseline_example4", "official_baseline", psd_mode="before")
    print_window_summary(baseline_window)
else:
    print("Skipping baseline window because data is not loaded.")
"""
    ),
    code(
        r"""
def plot_window_timeseries(window: dict, filename: str) -> None:
    plt.figure(figsize=(10, 4))
    for i, name in enumerate(channel_names):
        plt.plot(window["data_time"] / DAY, window["data_channels_td"][i], lw=0.8, label=name)
    plt.axvline(window["tc_day"], color="k", ls="--", lw=1.0, label="coalescence")
    plt.xlabel("Time (day)")
    plt.ylabel("TDI")
    plt.title(window["label"])
    plt.legend()
    save_current_figure(filename)

def plot_window_frequency(window: dict, filename: str) -> None:
    plt.figure(figsize=(10, 4))
    for i, name in enumerate(channel_names):
        plt.loglog(window["data_frequency"], np.abs(window["data_channels_fd"][i]), label=f"{name} data")
        plt.loglog(window["data_frequency"], np.sqrt(window["psd_channels"][i] * window["Tobs"] / 2.0), ls="--", label=f"{name} noise level")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("TDI (1/Hz)")
    plt.title(window["label"])
    plt.legend(ncol=2)
    save_current_figure(filename)

if baseline_window is not None:
    plot_window_timeseries(baseline_window, "01_baseline_timeseries.png")
    plot_window_frequency(baseline_window, "02_baseline_frequency_psd.png")
else:
    print("Baseline plots skipped.")
"""
    ),
    md("## 9. Model Setup Shared by Both Runs"),
    code(
        r"""
orbit = None
WFG = None
FDTDI = None

if ORBIT_DIR.exists() and (baseline_window is not None or DATA_DIR.exists() or PARAM_DIR.exists()):
    orbit = Orbit(OrbitDir=str(ORBIT_DIR))
    WFG = WaveformGeneratorFRef(mode="primary")
    FDTDI = FDTDIResponseGeneratorFRef(orbit_class=orbit, waveform_generator=WFG)
    print("Orbit, WF4Py waveform generator, and FDTDI response generator initialized.")
else:
    print("Model setup skipped because orbit path or data files are unavailable.")

def build_response_kwargs(window: dict, interpolation_method="cubic") -> dict:
    return dict(fmin=FMIN, fmax=FMAX, fref=1e-3, modes=[(2, 2)], tmin=window["data_time"][0]/DAY, tmax=window["data_time"][-1]/DAY, tref_at_constellation=True, TDIGeneration="2nd", optimal_combination=True, drop_T=True, interpolation_method=interpolation_method)
"""
    ),
    md(
        """
## 10. F-statistics Search and Waveform Reconstruction

This migrates official Example 4 cells 18--30.
"""
    ),
    code(
        r"""
def build_intrinsic_priors(response_kwargs_direct: dict) -> np.ndarray:
    return np.array([[5.0, 7.0], [0.01, 0.99], [-0.9, 0.9], [-0.9, 0.9], [response_kwargs_direct["tmin"], response_kwargs_direct["tmax"]], [0.0, TWOPI], [-1.0, 1.0]])

def run_fstat_search(window: dict, maxiter: int = 1000, popsize_factor: int = 5) -> dict:
    if FDTDI is None:
        raise RuntimeError("Initialize FDTDI before running F-statistics search.")
    response_kwargs_interp = build_response_kwargs(window, interpolation_method="cubic")
    response_kwargs_direct = response_kwargs_interp.copy()
    response_kwargs_direct["interpolation_method"] = None
    intrinsic_param_priors = build_intrinsic_priors(response_kwargs_direct)
    Fstat = FstatisticsFref(response_generator=FDTDI, frequency=window["data_frequency"], data=window["data_channels_fd"], invserse_covariance_matrix=window["InvCovMat"], response_parameters=response_kwargs_interp, use_gpu=False)

    def cost_function(norm_int_params):
        try:
            int_params = norm_int_params * (intrinsic_param_priors[:, 1] - intrinsic_param_priors[:, 0]) + intrinsic_param_priors[:, 0]
            return -Fstat.calculate_Fstat(intrinsic_parameters=Fstat.IntParamArr2ParamDict(int_params))
        except np.linalg.LinAlgError:
            return np.inf

    n_dim_int = 7
    bounds = np.array([np.zeros(n_dim_int), np.ones(n_dim_int)]).T
    DE_result = differential_evolution(func=cost_function, bounds=bounds, x0=None, strategy="best1exp", maxiter=maxiter, popsize=popsize_factor*n_dim_int, tol=1e-6, atol=1e-8, mutation=(0.4, 0.95), recombination=0.7, disp=True, polish=False, workers=-1)
    searched_int_params = Fstat.IntParamArr2ParamDict(DE_result.x * (intrinsic_param_priors[:, 1] - intrinsic_param_priors[:, 0]) + intrinsic_param_priors[:, 0])
    searched_a = Fstat.calculate_Fstat(intrinsic_parameters=searched_int_params, return_a=True)
    searched_parameters = dict(searched_int_params, **Fstat.a_to_extrinsic(searched_a))
    searched_wf = FDTDI.Response(searched_parameters, window["data_frequency"], **response_kwargs_interp)
    searched_parameters_reflected = get_reflected_parameter_dict_Fref(searched_params=searched_parameters, orbit=orbit)
    searched_wf_reflected = FDTDI.Response(parameters=searched_parameters_reflected, freqs=window["data_frequency"], **response_kwargs_interp)
    save_parameter_dict(searched_parameters, f"{window['label']}_searched_parameters.json")
    save_parameter_dict(searched_parameters_reflected, f"{window['label']}_searched_parameters_reflected.json")
    return dict(Fstat=Fstat, DE_result=DE_result, searched_parameters=searched_parameters, searched_parameters_reflected=searched_parameters_reflected, searched_wf=searched_wf, searched_wf_reflected=searched_wf_reflected, response_kwargs_interp=response_kwargs_interp, response_kwargs_direct=response_kwargs_direct, intrinsic_param_priors=intrinsic_param_priors)

def plot_reconstruction(window: dict, search: dict, reflected: bool, filename: str) -> None:
    wf_key = "searched_wf_reflected" if reflected else "searched_wf"
    title = "reflected" if reflected else "direct"
    plt.figure(figsize=(12, 5))
    for i, name in enumerate(channel_names):
        plt.subplot(1, 2, i+1)
        plt.loglog(window["data_frequency"], np.abs(window["data_channels_fd"][i]), label=f"{name} data", color=BLUE, lw=3, alpha=0.5)
        plt.loglog(window["data_frequency"], np.abs(search[wf_key][i]), label=f"{name} reconstructed", color=RED, lw=1, ls="--")
        plt.loglog(window["data_frequency"], np.abs(window["data_channels_fd"][i] - search[wf_key][i]), label=f"{name} residual", color="grey", lw=1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("TDI (1/Hz)")
        plt.ylim(1e-21, 1e-16)
        plt.legend(loc="upper left")
    plt.suptitle(f"{window['label']} reconstruction ({title})")
    save_current_figure(filename)
"""
    ),
    code(
        r"""
baseline_search = None
if RUN_BASELINE_SEARCH:
    baseline_search = run_fstat_search(baseline_window)
    plot_reconstruction(baseline_window, baseline_search, reflected=False, filename="03_baseline_reconstruction_direct.png")
    plot_reconstruction(baseline_window, baseline_search, reflected=True, filename="04_baseline_reconstruction_reflected.png")
else:
    print("RUN_BASELINE_SEARCH=False. Official baseline search code is present but not executed.")
"""
    ),
    md(
        """
## 11. Fisher Analysis

This migrates official Example 4 cells 31--36.
"""
    ),
    code(
        r"""
def run_fisher_analysis(window: dict, search: dict) -> MultiChannelFisher:
    def fisher_waveform_wrapper(param_dict, frequencies):
        return FDTDI.Response(parameters=param_dict, freqs=np.array(frequencies), **search["response_kwargs_interp"])
    analyze_param_step_dict = {"chirp_mass": -10.0, "mass_ratio": -0.01, "spin_1z": -0.01, "spin_2z": -0.01, "reference_time": -0.001, "reference_phase": -0.01, "luminosity_distance": -10.0, "inclination": -0.01, "longitude": -0.01, "latitude": -0.01, "psi": -0.01}
    FIM = MultiChannelFisher(waveform_generator=fisher_waveform_wrapper, param_dict=search["searched_parameters"], analyze_param_step_dict=analyze_param_step_dict, frequency=window["data_frequency"], inverse_covariance=window["InvCovMat"], verbose=0)
    FIM.auto_test_step()
    FIM.calculate_Fisher()
    FIM.calculate_errors()
    pd.DataFrame([{"parameter": k, "fim_error": v} for k, v in FIM.param_errors.items()]).to_csv(RESULT_DIR / f"{window['label']}_fisher_errors.csv", index=False)
    return FIM

def compare_search_to_injection(window: dict, search: dict, FIM) -> pd.DataFrame:
    injected_parameters_fref = injected_parameters.copy()
    injected_parameters_fref.pop("coalescence_time", None)
    injected_parameters_fref.pop("coalescence_phase", None)
    injected_parameters_fref["reference_time"] = None
    injected_parameters_fref["reference_phase"] = None
    rows = []
    for key, truth in injected_parameters_fref.items():
        if truth is None or key not in search["searched_parameters"]:
            continue
        rows.append(dict(parameter=key, injected=truth, searched=search["searched_parameters"][key], searched_abs_error=abs(truth-search["searched_parameters"][key]), reflected=search["searched_parameters_reflected"].get(key), reflected_abs_error=abs(truth-search["searched_parameters_reflected"][key]) if key in search["searched_parameters_reflected"] else np.nan, fim_error=FIM.param_errors.get(key, np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / f"{window['label']}_search_vs_injection.csv", index=False)
    return df

baseline_FIM = None
baseline_search_comparison = None
if baseline_search is not None:
    baseline_FIM = run_fisher_analysis(baseline_window, baseline_search)
    baseline_search_comparison = compare_search_to_injection(baseline_window, baseline_search, baseline_FIM)
    display(baseline_search_comparison)
else:
    print("Fisher analysis waiting for baseline_search.")
"""
    ),
    md(
        """
## 12. Heterodyned Likelihood, Priors, and Nested Sampling

This migrates official Example 4 cells 37--47. The default sampler is NESSAI.
"""
    ),
    code(
        r"""
class BilbyLikelihoodWrapper(bilby.Likelihood):
    def __init__(self, like_object, like_type="heterodyned"):
        super().__init__(parameters={"chirp_mass": None, "mass_ratio": None, "spin_1z": None, "spin_2z": None, "reference_time": None, "reference_phase": None, "luminosity_distance": None, "inclination": None, "longitude": None, "latitude": None, "psi": None})
        self.like_object = like_object
        self.like_type = like_type
    def log_likelihood(self):
        parameter_array = ParamDict2ParamArrFref(self.parameters)
        if self.like_type == "heterodyned":
            return self.like_object.het_log_like(parameter_array=parameter_array)
        return self.like_object.full_log_like(parameter_array=parameter_array)

def build_likelihood(window: dict, search: dict) -> Likelihood:
    Like = Likelihood(response_generator=FDTDI, frequency=window["data_frequency"], data=window["data_channels_fd"], invserse_covariance_matrix=window["InvCovMat"], response_parameters=search["response_kwargs_direct"], Fref_waveform=True, use_gpu=False)
    Like.prepare_het_log_like(base_parameters=ParamDict2ParamArrFref(search["searched_parameters"]))
    return Like

def build_priors(search: dict, FIM) -> bilby.core.prior.PriorDict:
    sp, pe = search["searched_parameters"], FIM.param_errors
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.prior.Uniform(sp["chirp_mass"]-5*pe["chirp_mass"], sp["chirp_mass"]+5*pe["chirp_mass"], name="chirp_mass", latex_label="$\\mathcal{M}_c$")
    priors["mass_ratio"] = bilby.prior.Uniform(max(0.1, sp["mass_ratio"]-5*pe["mass_ratio"]), min(0.99, sp["mass_ratio"]+5*pe["mass_ratio"]), name="mass_ratio", latex_label="$q$")
    priors["spin_1z"] = bilby.prior.Uniform(max(-0.9, sp["spin_1z"]-5*pe["spin_1z"]), min(0.9, sp["spin_1z"]+5*pe["spin_1z"]), name="spin_1z", latex_label="$\\chi_{z1}$")
    priors["spin_2z"] = bilby.prior.Uniform(max(-0.9, sp["spin_2z"]-5*pe["spin_2z"]), min(0.9, sp["spin_2z"]+5*pe["spin_2z"]), name="spin_2z", latex_label="$\\chi_{z2}$")
    priors["reference_time"] = bilby.prior.Uniform(sp["reference_time"]-5*pe["reference_time"], sp["reference_time"]+5*pe["reference_time"], name="reference_time", latex_label="$t_\\mathrm{ref}$")
    priors["reference_phase"] = bilby.prior.Uniform(0.0, TWOPI, name="reference_phase", latex_label="$\\varphi_\\mathrm{ref}$", boundary="periodic")
    priors["luminosity_distance"] = bilby.prior.Uniform(max(6e3, sp["luminosity_distance"]-5*pe["luminosity_distance"]), min(1e5, sp["luminosity_distance"]+5*pe["luminosity_distance"]), name="luminosity_distance", latex_label="$d_L$")
    priors["inclination"] = bilby.prior.Sine(0.0, PI, name="inclination", latex_label="$\\iota$")
    priors["longitude"] = bilby.prior.Uniform(0.0, TWOPI, name="longitude", latex_label="$\\lambda$", boundary="periodic")
    priors["latitude"] = bilby.prior.Cosine(-PI/2.0, PI/2.0, name="latitude", latex_label="$\\beta$")
    priors["psi"] = bilby.prior.Uniform(0.0, PI, name="psi", latex_label="$\\psi$", boundary="periodic")
    return priors

def build_injected_parameters_fref() -> dict:
    injected_parameters_fref = injected_parameters.copy()
    injected_parameters_fref.pop("coalescence_time", None)
    injected_parameters_fref.pop("coalescence_phase", None)
    injected_parameters_fref["reference_time"] = None
    injected_parameters_fref["reference_phase"] = None
    return injected_parameters_fref

PARAMETERS_TO_COMPARE = ["chirp_mass", "mass_ratio", "spin_1z", "spin_2z", "reference_time", "reference_phase", "luminosity_distance", "inclination", "longitude", "latitude", "psi"]

def run_nested_sampler(window: dict, search: dict, FIM, label: str, smoke_test: bool = True):
    if not HAS_NESSAI:
        raise RuntimeError("NESSAI is not installed/importable. Install nessai before running the official sampler.")
    Like = build_likelihood(window, search)
    result = bilby.run_sampler(likelihood=BilbyLikelihoodWrapper(Like), priors=build_priors(search, FIM), npool=os.cpu_count(), injection_parameters=build_injected_parameters_fref(), outdir=str(RESULT_DIR / f"{label}_samples"), label=label, plot=True, resume=False, **(SMOKE_TEST_SAMPLER_SETTINGS if smoke_test else OFFICIAL_SAMPLER_SETTINGS))
    result.plot_corner(save=True)
    summary = posterior_summary(result.posterior, PARAMETERS_TO_COMPARE)
    summary.to_csv(RESULT_DIR / f"{label}_posterior_summary.csv", index=False)
    return result, summary

baseline_result = None
baseline_summary = None
if RUN_BASELINE_SAMPLER:
    baseline_result, baseline_summary = run_nested_sampler(baseline_window, baseline_search, baseline_FIM, label="baseline_example4", smoke_test=USE_SMOKE_TEST_SAMPLER)
else:
    print("RUN_BASELINE_SAMPLER=False. Sampler code is present but not executed.")
"""
    ),
    md("## 13. Build Task-Required 5-Day Window"),
    code(
        r"""
five_day_window = None
if read_dict is not None:
    five_day_window = build_window_data("task_five_day", "task_five_day", psd_mode="before")
    print_window_summary(five_day_window)
    plot_window_timeseries(five_day_window, "05_five_day_timeseries.png")
    plot_window_frequency(five_day_window, "06_five_day_frequency_psd.png")
else:
    print("Skipping 5-day window because data is not loaded.")
"""
    ),
    md("## 14. Run Search, Fisher, Likelihood, and Sampler on the 5-Day Window"),
    code(
        r"""
five_day_search = None
five_day_FIM = None
five_day_search_comparison = None
if RUN_FIVE_DAY_SEARCH:
    five_day_search = run_fstat_search(five_day_window)
    plot_reconstruction(five_day_window, five_day_search, reflected=False, filename="07_five_day_reconstruction_direct.png")
    plot_reconstruction(five_day_window, five_day_search, reflected=True, filename="08_five_day_reconstruction_reflected.png")
    five_day_FIM = run_fisher_analysis(five_day_window, five_day_search)
    five_day_search_comparison = compare_search_to_injection(five_day_window, five_day_search, five_day_FIM)
    display(five_day_search_comparison)
else:
    print("RUN_FIVE_DAY_SEARCH=False. Modified-window search code is present but not executed.")

five_day_result = None
five_day_summary = None
if RUN_FIVE_DAY_SAMPLER:
    five_day_result, five_day_summary = run_nested_sampler(five_day_window, five_day_search, five_day_FIM, label="task_five_day", smoke_test=USE_SMOKE_TEST_SAMPLER)
else:
    print("RUN_FIVE_DAY_SAMPLER=False. Modified-window sampler code is present but not executed.")
"""
    ),
    md("## 15. Baseline vs Modified-Window Comparison"),
    code(
        r"""
def compare_summaries(baseline_summary: pd.DataFrame, five_day_summary: pd.DataFrame) -> pd.DataFrame:
    base = baseline_summary.add_prefix("baseline_").rename(columns={"baseline_parameter": "parameter"})
    five = five_day_summary.add_prefix("five_day_").rename(columns={"five_day_parameter": "parameter"})
    merged = pd.merge(base, five, on="parameter", how="outer")
    merged["ci90_width_ratio_5day_over_baseline"] = merged["five_day_ci90_width"] / merged["baseline_ci90_width"]
    return merged

comparison = pd.DataFrame(columns=["parameter", "baseline_median", "baseline_ci90_low", "baseline_ci90_high", "baseline_ci90_width", "five_day_median", "five_day_ci90_low", "five_day_ci90_high", "five_day_ci90_width", "ci90_width_ratio_5day_over_baseline"])
if baseline_summary is not None and five_day_summary is not None:
    comparison = compare_summaries(baseline_summary, five_day_summary)
    comparison.to_csv(RESULT_DIR / "baseline_vs_five_day_parameter_summary.csv", index=False)
display(comparison)
"""
    ),
    md("## 16. Taiji-Frame Sky Position Plot"),
    code(
        r"""
def plot_taiji_frame_position(result, label: str, filename: str) -> None:
    num_sample = len(result.posterior["longitude"])
    longitude_TJ = np.zeros(num_sample)
    latitude_TJ = np.zeros(num_sample)
    for i in range(num_sample):
        lon, lat, _ = SSBPosToDetectorFrame(lon_ssb=result.posterior["longitude"][i], lat_ssb=result.posterior["latitude"][i], psi_ssb=result.posterior["psi"][i], orbit_time_SI=injected_parameters["coalescence_time"]*DAY, orbit=orbit)
        longitude_TJ[i] = lon % TWOPI
        latitude_TJ[i] = lat
    plt.figure(figsize=(5.2, 4.4))
    plt.hist2d(x=longitude_TJ, y=latitude_TJ, bins=50)
    plt.xlabel("longitude (rad)")
    plt.ylabel("latitude (rad)")
    plt.title(label)
    save_current_figure(filename)

if baseline_result is not None:
    plot_taiji_frame_position(baseline_result, "baseline Taiji-frame sky position", "09_baseline_taiji_frame_sky.png")
if five_day_result is not None:
    plot_taiji_frame_position(five_day_result, "five-day Taiji-frame sky position", "10_five_day_taiji_frame_sky.png")
if baseline_result is None and five_day_result is None:
    print("Sky-position plots waiting for sampler results.")
"""
    ),
    md("## 17. Result Manifest for README"),
    code(
        r"""
manifest = {
    "baseline_timeseries": "figures/task5_subtask2/01_baseline_timeseries.png",
    "baseline_frequency_psd": "figures/task5_subtask2/02_baseline_frequency_psd.png",
    "baseline_reconstruction_direct": "figures/task5_subtask2/03_baseline_reconstruction_direct.png",
    "baseline_reconstruction_reflected": "figures/task5_subtask2/04_baseline_reconstruction_reflected.png",
    "five_day_timeseries": "figures/task5_subtask2/05_five_day_timeseries.png",
    "five_day_frequency_psd": "figures/task5_subtask2/06_five_day_frequency_psd.png",
    "five_day_reconstruction_direct": "figures/task5_subtask2/07_five_day_reconstruction_direct.png",
    "five_day_reconstruction_reflected": "figures/task5_subtask2/08_five_day_reconstruction_reflected.png",
    "baseline_taiji_frame_sky": "figures/task5_subtask2/09_baseline_taiji_frame_sky.png",
    "five_day_taiji_frame_sky": "figures/task5_subtask2/10_five_day_taiji_frame_sky.png",
    "comparison_table": "results/task5_subtask2/baseline_vs_five_day_parameter_summary.csv",
}
save_json(manifest, "manifest.json")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
"""
    ),
    md(
        """
## 18. Final Discussion Notes

Complete this section after the full runs finish.

Report these points in the README:

1. Whether official Example 4 was reproduced.
2. Baseline window: `tc - 2.5 days` to `tc + 2.5 days`.
3. Modified task window: `tc - 4 days` to `tc + 1 day`.
4. Whether the modified window changes search parameters, reconstruction residuals, Fisher estimates, or posterior credible intervals.
5. Which parameters improve most and which remain degenerate or multimodal.
6. Limitations: idealized single-bright-MBHB assumption, simplified noise treatment, possible multimodality, and local Windows environment constraints.

Conclusion draft:

- Baseline reproduction: TODO after TDC data and sampler run.
- Modified 5-day run: TODO after TDC data and sampler run.
- Quantitative posterior comparison: TODO after both posterior summaries are generated.
"""
    ),
]

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (task5-tdc)",
        "language": "python",
        "name": "task5-tdc",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nb.cells = cells
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH} with {len(cells)} cells")
