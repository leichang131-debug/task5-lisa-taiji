"""Short timing benchmark for the official GPU route."""

from __future__ import annotations

import time

import cupy as xp
import numpy as np

from gpu_subtask2_preflight import (
    DAY,
    DATA_DIR,
    ORBIT_DIR,
    PARAM_DIR,
    build_window,
    h5py,
    read_dict_from_h5,
    Orbit,
    BBHxWaveformGenerator,
    BBHxFDTDIResponseGenerator,
    Likelihood,
    ParamDict2ParamArr,
)


def median_time(fn, n: int = 10) -> tuple[float, float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.mean(times))


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
param_arr = np.asarray(ParamDict2ParamArr(injected_parameters), dtype=float)
like.prepare_het_log_like(base_parameters=param_arr)


def waveform_once():
    return fdtdi.Response(parameters=injected_parameters, freqs=frequency, **response_kwargs_interp)


def heterodyne_once():
    return like.het_log_like(parameter_array=param_arr)


def heterodyne_vectorized_4000():
    samples = np.tile(param_arr[:, None], (1, 4000))
    samples += np.random.default_rng(1234).normal(scale=1e-8, size=samples.shape)
    return like.het_log_like_vectorized(samples)


print("frequency_bins", int(frequency.shape[0]))
print("gpu_device_count", int(xp.cuda.runtime.getDeviceCount()))
print("waveform_once_median_mean_sec", median_time(waveform_once, n=10))
print("heterodyne_once_median_mean_sec", median_time(heterodyne_once, n=20))
print("heterodyne_vectorized_4000_median_mean_sec", median_time(heterodyne_vectorized_4000, n=5))
