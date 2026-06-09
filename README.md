# task5-lisa-taiji

UCAS 2026 Innovation Practice Task 5: LISA Data Challenge time-frequency visualization and Taiji MBHB parameter estimation.

本仓库用于整理并完成中国科学院大学 2026 年科创计划任务五：LISA Data Challenge / Taiji Data Challenge。项目分为两个子任务：

1. LISA Data Challenge 数据时频变换及可视化。
2. Taiji Data Challenge 大质量双黑洞信号参数估计。

## Repository Structure

```text
task5-lisa-taiji/
+-- README.md
+-- environment.yml
+-- requirements.txt
+-- notebooks/
|   +-- 01_ldc_time_frequency_visualization.ipynb
|   +-- 02_taiji_mbhb_parameter_estimation.ipynb
+-- src/
|   +-- __init__.py
|   +-- data_utils.py
|   +-- plotting.py
|   +-- frft_utils.py
+-- figures/
    +-- task5_subtask1/
    +-- task5_subtask2/
```

## Data

The LDC HDF5 file is not stored in this repository because it is large and should be downloaded separately.

Required file for subtask 1:

```text
LDC2_spritz_mbhb1_training_v1.h5
```

Original task download information:

```text
Baidu Netdisk: https://pan.baidu.com/s/1bfAfIgvi9Gfhlf3SIgrdUw
Extraction code: ucas
```

After downloading, set the local path in the first notebook:

```python
DATA_PATH = r"E:\BaiduNetdiskDownload\LDC2_spritz_mbhb1_training_v1.h5"
```

Large local data files are ignored by `.gitignore`.

## Subtask 1: LDC Time-Frequency Visualization

Notebook:

```text
notebooks/01_ldc_time_frequency_visualization.ipynb
```

Main goals:

1. Read LISA Data Challenge HDF5 data with `h5py`.
2. Replace NaN values with zero as required by the task statement.
3. Inspect the HDF5 group and dataset structure.
4. Use the observed A-like TDI channel from `obs/tdi` as the main analysis stream.
5. Plot raw time-series and frequency-domain diagnostic figures.
6. Perform Wilson-Daubechies-Meyer wavelet transform with `WDMWaveletTransforms`.
7. Perform fractional Fourier transform with the NumPy/SciPy helper in `src/frft_utils.py`.
8. Save clear time-frequency visualization figures under `figures/task5_subtask1/`.

FRFT reference evaluation:

- `siddharth-maddali/frft`: useful multi-dimensional FRFT reference based on the Ozaktas chirp-convolution formulation.
- `nanaln/python_frft`: selected as the main reference because it implements a one-dimensional NumPy/SciPy fast FRFT, matching the TDI time-series data.
- `MStamatis/frft2d`: useful for two-dimensional data, but not used as the main method because this subtask transforms one-dimensional TDI time series.

### Subtask 1 Current Results

The current subtask 1 notebook has been executed successfully with the real HDF5 file. The main analysis uses `obs/tdi` after filling NaN values with zero. The `sky/tdi` stream is used only as a reference to verify the signal location.

Key data checks:

- HDF5 file: `LDC2_spritz_mbhb1_training_v1.h5`
- Main dataset: `obs/tdi`
- Reference dataset: `sky/tdi`
- Sample cadence: `dt = 5.0 s`
- NaN counts before zero filling: `X = 10080`, `Y = 10080`, `Z = 10080`

Quantitative sanity checks from the executed notebook:

| Check | Value | Interpretation |
| --- | ---: | --- |
| WDM low-frequency coalescence/background mean energy ratio | 20.58 | The observed TDI stream has clearly enhanced WDM energy near coalescence. |
| WDM low-frequency peak time relative to coalescence | 0.035 days | The WDM peak occurs close to the catalog coalescence time. |
| FRFT best near/quiet peak-energy ratio | 15.14 | The near-coalescence segment is more concentrated in FRFT space than an earlier quiet segment. |
| FRFT best alpha for near/quiet contrast | 0.24 | Fractional order where the near-coalescence contrast is strongest in the scan. |

Generated figures:

- [Observed and signal-only A channel near coalescence](figures/task5_subtask1/01_obs_and_sky_a_timeseries_relative_to_tc.png)
- [Frequency-domain diagnostic for `obs/tdi A`](figures/task5_subtask1/02_obs_a_frequency_diagnostic.png)
- [Cropped observed/reference window](figures/task5_subtask1/03_cropped_obs_and_sky_a_window.png)
- [WDM transform, `obs/tdi A`, `Nf=256`](figures/task5_subtask1/04_wdm_obs_a_nfreq_256.png)
- [WDM transform, `obs/tdi A`, `Nf=512`](figures/task5_subtask1/04_wdm_obs_a_nfreq_512.png)
- [WDM transform, `obs/tdi A`, `Nf=1024`](figures/task5_subtask1/04_wdm_obs_a_nfreq_1024.png)
- [Reference WDM transform, `sky/tdi A`, `Nf=512`](figures/task5_subtask1/04_wdm_sky_reference_nfreq_512.png)
- [Observed segment used for FRFT](figures/task5_subtask1/05_frft_obs_input_segment.png)
- [FRFT alpha scan, `obs/tdi A`](figures/task5_subtask1/06_frft_obs_alpha_scan.png)

Interpretation:

The WDM plots show a low-frequency energy enhancement near the catalog coalescence time. The `obs/tdi` result is noisier than the `sky/tdi` reference, as expected for the observed stream, but the enhancement remains visible and is supported by the energy-ratio check. The FRFT scan also shows stronger near-coalescence concentration than an earlier quiet segment, supporting the use of FRFT as a complementary transform.

## Subtask 2: Taiji MBHB Parameter Estimation

Notebook:

```text
notebooks/02_taiji_mbhb_parameter_estimation.ipynb
```

Main goals:

1. Install and run Triangle-BBH following its README.
2. Reproduce Example 4 data handling as the baseline result.
3. Understand TDI data, Fourier transform, massive black-hole binary waveform, and Bayesian inference.
4. Extend the parameter-estimation data window to 5 days: 4 days before `coalescence_time` and 1 day after it.
5. Keep the required NESSAI nested-sampling stage, but accelerate it with the GPU BBHx heterodyned likelihood used by the official GPU examples.
6. Save evidence summaries, posterior tables, corner plots, trace/diagnostic plots, and parameter comparison tables under `figures/task5_subtask2/` and `results/task5_subtask2/`.

Official Example 4 data files:

- `0_2_MBHB_TDIXYZ.h5`: https://zenodo.org/records/15469724/files/0_2_MBHB_TDIXYZ.h5?download=1
- `0_2_MBHB_parameters.h5`: https://zenodo.org/records/15532090/files/0_2_MBHB_parameters.h5?download=1

The subtask 2 notebook first reproduces the official symmetric baseline window, `tc - 2.5 days` to `tc + 2.5 days`, and then rebuilds the full pipeline for the required asymmetric 5-day window, `tc - 4 days` to `tc + 1 day`.

Current main route:

```text
Example 4 data loading / A,E construction / FFT / PSD / covariance / windows
+ Example 5 BBHx GPU F-statistics search
+ Example 2-style GPU heterodyned likelihood
+ vectorised nessai.FlowSampler nested sampling
```

The CPU Bilby/NESSAI path is retained in the notebook as a reference fallback, but it is not the default route because the local run is much slower. The previous CPU NESSAI attempt ran for about 4 hours and still had `dlogZ ~ 1497`, far from the `stopping=0.1` target. The current route keeps NESSAI but exposes `Triangle_BBH.Fisher.Likelihood.het_log_like_vectorized` through a custom `nessai.model.Model`, so NESSAI can evaluate batches of live points with the GPU heterodyned likelihood. This should be substantially faster than Example 4's full-frequency CPU likelihood if the benchmark cell confirms high batch throughput.

Four guardrails are built into the notebook before production sampling and interpretation:

- Coordinate check: `*_gpu_nessai_coordinate_check.json` compares the likelihood from Triangle-BBH internal `ParamDict2ParamArr` coordinates against an intentionally wrong raw physical-coordinate call. Production runs should proceed only if the internal-coordinate likelihood is the self-consistent branch and the round-trip error is negligible.
- Sky-prior mode: `GPU_NESSAI_SKY_PRIOR_MODE = "local_fisher"` is the default because the heterodyned likelihood is built around one fiducial waveform. `wide_sky` is available only as an exploratory stress test; for a rigorous direct-vs-reflected comparison, run two local branches with separate fiducial/search points and compare their `logZ`.
- NESSAI insertion-index diagnostic: `*_gpu_nessai_native_diagnostics.json` lists NESSAI's native diagnostic/log files, and `*_gpu_nessai_diagnostics.json` records a lightweight secondary KS screen. Prefer the native insertion-index diagnostics when available; the JSON KS p-value is only a quick warning signal.
- Fisher-box boundary diagnostic: `*_gpu_nessai_boundary_check.csv` records the posterior fraction near each Fisher-centred sampling-box bound, not the global physical hard bounds. If posterior mass touches a sampling-box edge in the full run, widen/recenter that parameter before interpreting credible intervals.

Current subtask 2 outputs:

- [Baseline time series](figures/task5_subtask2/01_baseline_timeseries.png)
- [Baseline frequency-domain data and PSD](figures/task5_subtask2/02_baseline_frequency_psd.png)
- [Baseline direct reconstruction](figures/task5_subtask2/03_baseline_reconstruction_direct.png)
- [Baseline reflected reconstruction](figures/task5_subtask2/04_baseline_reconstruction_reflected.png)
- [Task 5-day time series](figures/task5_subtask2/05_five_day_timeseries.png)
- [Task 5-day frequency-domain data and PSD](figures/task5_subtask2/06_five_day_frequency_psd.png)
- [Task 5-day direct reconstruction](figures/task5_subtask2/07_five_day_reconstruction_direct.png)
- [Task 5-day reflected reconstruction](figures/task5_subtask2/08_five_day_reconstruction_reflected.png)
- [Baseline Taiji-frame sky check](figures/task5_subtask2/09_baseline_taiji_frame_sky.png)
- [Task 5-day Taiji-frame sky check](figures/task5_subtask2/10_five_day_taiji_frame_sky.png)

Current quantitative summaries:

- [Baseline search vs injection](results/task5_subtask2/baseline_example4_search_vs_injection.csv)
- [Task 5-day search vs injection](results/task5_subtask2/task_five_day_search_vs_injection.csv)
- [Baseline posterior summary](results/task5_subtask2/baseline_example4_posterior_summary.csv)
- [Task 5-day posterior summary](results/task5_subtask2/task_five_day_posterior_summary.csv)
- [Baseline vs 5-day posterior comparison](results/task5_subtask2/baseline_vs_five_day_parameter_summary.csv)
- [Baseline GPU preflight](results/task5_subtask2/baseline_example4_gpu_preflight.json)
- [Baseline GPU search vs injection](results/task5_subtask2/baseline_example4_gpu_search_vs_injection.csv)
- [Baseline GPU reflected search parameters](results/task5_subtask2/baseline_example4_gpu_searched_parameters_reflected.json)
- Baseline GPU NESSAI outputs: `baseline_example4_gpu_nessai_*` under `results/task5_subtask2/` after running notebook section 18b, including posterior, evidence, insertion-index diagnostics, and boundary checks.
- [Baseline GPU Eryn posterior summary](results/task5_subtask2/baseline_example4_gpu_eryn_posterior_summary.csv)
- [Task 5-day GPU preflight](results/task5_subtask2/task_five_day_gpu_preflight.json)
- [Task 5-day GPU search vs injection](results/task5_subtask2/task_five_day_gpu_search_vs_injection.csv)
- [Task 5-day GPU reflected search parameters](results/task5_subtask2/task_five_day_gpu_searched_parameters_reflected.json)
- Task 5-day GPU NESSAI outputs: `task_five_day_gpu_nessai_*` under `results/task5_subtask2/` after running notebook section 19, including posterior, evidence, insertion-index diagnostics, and boundary checks.
- [Task 5-day GPU Eryn posterior summary](results/task5_subtask2/task_five_day_gpu_eryn_posterior_summary.csv)
- [Baseline vs 5-day GPU Eryn comparison](results/task5_subtask2/baseline_vs_five_day_gpu_eryn_parameter_summary.csv)

Subtask 2 interpretation:

The real TDC II files were loaded from `0_2_MBHB_TDIXYZ.h5` and `0_2_MBHB_parameters.h5`. Both the official baseline window and the required asymmetric 5-day window contain 43,201 samples at `dt = 10 s`, use A/E channels after the XYZ-to-AET conversion, and keep the same frequency band, `5e-5 Hz <= f <= 1e-2 Hz`.

The 100-iteration validation F-statistics search recovers the intrinsic parameters at a useful level for notebook verification: baseline chirp-mass error is about `7.31e3 Msun`, mass-ratio error is `2.80e-4`, and spin errors are `7.98e-3` and `2.21e-2`; for the required 5-day window, chirp-mass error is about `1.15e4 Msun`, mass-ratio error is `3.74e-3`, and spin errors are `6.27e-3` and `3.42e-2`. The sky/extrinsic maximum currently lands on the ecliptic-plane reflected branch rather than the injected sky branch. The notebook therefore writes both the direct search parameters and the `Triangle_BBH.Utils.get_reflected_parameter_dict` reflected parameters, and the search-vs-injection CSV files include `reflected` and `reflected_abs_error` columns for every comparable parameter. The direct and reflected reconstruction figures should be inspected together.

The GPU preflight JSON files now include `log_likelihood_at_injection` / `heterodyned_log_likelihood_at_injection`, computed with the same heterodyned likelihood object used by the GPU NESSAI route. These fields are intended as a sanity check that the likelihood can be evaluated at the injected parameters before long sampling starts.

The previous Windows-local posterior run used `bilby==1.0.0` with `dynesty==1.0.1` as a smoke sampler. Those CSV files are kept only as historical pipeline-validation outputs. The previous WSL2 GPU Eryn quick check is also retained only as a pipeline-validation posterior. The production route is now NESSAI with `GPU_NESSAI_RUN_MODE = "pilot"` for a small validation run and `GPU_NESSAI_RUN_MODE = "full"` for the evidence-producing run. The NESSAI paper finds that gravitational-wave inference needs at least about 1000 live points and recommends 2000 for complex GW problems, so the `nlive=200` pilot must only be used to check the coordinate path, benchmark, and file outputs; scientific evidence and posterior statements should use `GPU_NESSAI_FULL_NLIVE = 2000`. The local benchmark currently evaluates `het_log_like_vectorized(4000)` in about `0.28 s`, i.e. roughly `1.3e4-1.4e4` likelihood points/s before NESSAI flow-training and pool-population overhead. The NESSAI benchmark JSON files should still be checked first; if batch likelihood throughput is poor, reduce `likelihood_chunksize` from `512` to `256`. Before a full run, confirm the notebook prints `PyTorch CUDA available: True`; otherwise NESSAI flow training may fall back to CPU and the runtime estimate can be too optimistic.

Because the current production route uses NESSAI nested sampling, it is expected to provide Bayesian evidence (`log Z`) and posterior samples for comparing the symmetric baseline and asymmetric 5-day windows. After the full run, Section 21 of the notebook should discuss NESSAI convergence/evidence uncertainty, insertion-index uniformity, posterior boundary contact, direct/reflected-sky degeneracy, and possible PSD-estimation differences between the two windows before reporting final posterior conclusions.

## Environment

For the lightweight subtask 1 environment, use:

```bash
pip install -r requirements.txt
```

For a conda-based environment, use:

```bash
conda env create -f environment.yml
conda activate task5-lisa-taiji
```

For the WSL2 GPU route used by subtask 2, use `environment-wsl-gpu.yml` as the reproducible dependency record and then compile BBHx from the official Triangle-BBH source inside WSL2. The local `tri_env` setup was verified with `cupy`, `bbhx`, `healpy`, `pycbc`, `nessai`, and `eryn`; CUDA compiler packages were pinned to the CUDA 12.6 series to match the current NVIDIA 12.7 driver. The helper script below recreates the `CUDAHOME` compatibility directory expected by BBHx:

```bash
bash scripts/setup_wsl_cuda_home.sh
```

The short GPU sanity check is:

```bash
python scripts/gpu_subtask2_preflight.py
```

The NESSAI vectorised-interface check is:

```bash
python scripts/gpu_nessai_vectorized_interface_check.py
```

Triangle-BBH and Triangle-Simulator may require a separate Linux or WSL2 environment. See the official repositories:

- Triangle-BBH: https://github.com/TriangleDataCenter/Triangle-BBH
- Triangle-Simulator: https://github.com/TriangleDataCenter/Triangle-Simulator

## Run Order

1. Run `notebooks/01_ldc_time_frequency_visualization.ipynb` for subtask 1.
2. Run `notebooks/02_taiji_mbhb_parameter_estimation.ipynb` with the `tri_env-task5-wsl2` kernel.
3. The notebook default main route now runs GPU preflight, GPU F-statistics search, GPU Fisher analysis, and vectorised GPU-heterodyned NESSAI for both the baseline and task 5-day windows.
4. Keep `RUN_CPU_EXAMPLE4_FSTAT = False` and `RUN_CPU_NESSAI = False` unless you explicitly want the slower CPU reference route.
5. Use `GPU_NESSAI_RUN_MODE = "pilot"` with `GPU_NESSAI_PILOT_NLIVE = 200` to validate the route quickly; switch to `GPU_NESSAI_RUN_MODE = "full"` with `GPU_NESSAI_FULL_NLIVE = 2000` for the evidence-producing run.
6. When switching from pilot to full, either clear the corresponding `results/task5_subtask2/*_gpu_nessai/` output directory or set `GPU_NESSAI_RESUME = False` for the first full run, then re-enable resume only for interrupted full runs.
7. Before the long run, check `python scripts/gpu_nessai_vectorized_interface_check.py`, the notebook-generated `*_gpu_nessai_coordinate_check.json`, and the `*_gpu_nessai_vectorized_benchmark.json` files.
8. After the full run, inspect `*_gpu_nessai_native_diagnostics.json`, `*_gpu_nessai_diagnostics.json`, and `*_gpu_nessai_boundary_check.csv` before interpreting `logZ`, CI widths, or baseline-vs-5-day differences.
9. For lower-level timing diagnostics only, run `python scripts/gpu_subtask2_benchmark.py`.

## Remaining Work

Subtask 2 is implemented with the GPU-heterodyned NESSAI route. The remaining production step is to run the notebook in `GPU_NESSAI_RUN_MODE = "pilot"` first, inspect the vectorised likelihood benchmark, coordinate checks, and file outputs, then run `GPU_NESSAI_RUN_MODE = "full"` for both windows. Evidence and posterior summaries should be treated as final only after insertion-index and boundary diagnostics pass.

Preliminary runtime estimate on the current WSL2 GPU setup:

- Cached-search pilot run (`nlive=200`, one window): about `10-30 min`, mostly NESSAI setup/training overhead rather than raw likelihood time; this is a pipeline check only.
- Cached-search full run (`nlive=2000`, one local-fisher window): plausibly `40-90 min` if vectorised throughput remains near the current benchmark and proposal efficiency is reasonable. If NESSAI flow training/pool population dominates, this may stretch toward `1-2 h`.
- Two windows full run: roughly `1.5-3 h`, with a cautious upper bound around `4 h` if flow-training efficiency is poor.
- Re-running the full GPU F-statistics search instead of using cached search results can add tens of minutes per window, depending on `GPU_SEARCH_MAXITER`.
- The post-run `*_gpu_nessai_evidence.json` records actual `wall_time_minutes`, `seed`, `nlive`, and `torch_cuda_available`; use those measured values in the final report rather than relying only on this preliminary estimate.
- `wide_sky` can be substantially slower and less reliable with a single heterodyned fiducial; prefer two separate local-fisher branches for direct/reflected mode comparison.

## Notes

This repository keeps third-party dependencies as external dependencies instead of copying their source code. Each third-party package follows its own license.
