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
2. Reproduce Example 4 as the baseline result.
3. Understand TDI data, Fourier transform, massive black-hole binary waveform, and Bayesian inference.
4. Extend the parameter-estimation data window to 5 days: 4 days before `coalescence_time` and 1 day after it.
5. Re-run the inference pipeline and compare the posterior results with the baseline.
6. Save corner plots, trace plots, and parameter comparison tables under `figures/task5_subtask2/`.

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

Triangle-BBH and Triangle-Simulator may require a separate Linux or WSL2 environment. See the official repositories:

- Triangle-BBH: https://github.com/TriangleDataCenter/Triangle-BBH
- Triangle-Simulator: https://github.com/TriangleDataCenter/Triangle-Simulator

## Run Order

1. Run `notebooks/01_ldc_time_frequency_visualization.ipynb` for subtask 1.
2. Clone and run the original Triangle-BBH Example 4 for baseline reproduction.
3. Run or adapt `notebooks/02_taiji_mbhb_parameter_estimation.ipynb` for the 5-day window experiment.
4. Update this README with final subtask 2 figures and conclusions.

## Remaining Work

Subtask 2 still needs to be completed:

- Baseline Example 4 result figures.
- Modified 5-day window result figures.
- Posterior corner plots.
- Parameter comparison table.
- Short analysis of how the 5-day window changes the inference result.

## Notes

This repository keeps third-party dependencies as external dependencies instead of copying their source code. Each third-party package follows its own license.
