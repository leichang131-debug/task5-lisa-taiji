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

Large local data and generated outputs are ignored by `.gitignore`.

## Subtask 1: LDC Time-Frequency Visualization

Notebook:

```text
notebooks/01_ldc_time_frequency_visualization.ipynb
```

Main goals:

1. Read LISA Data Challenge HDF5 data with `h5py`.
2. Replace NaN values with zero as required by the task statement.
3. Inspect the HDF5 group and dataset structure.
4. Select a TDI channel such as `X`, `A`, or `E` for analysis.
5. Plot raw time-series and frequency-domain diagnostic figures.
6. Perform Wilson-Daubechies-Meyer wavelet transform with `WDMWaveletTransforms`.
7. Perform fractional Fourier transform with the NumPy/SciPy helper in `src/frft_utils.py`.
8. Save clear time-frequency visualization figures under `figures/task5_subtask1/`.

FRFT reference evaluation:

- `siddharth-maddali/frft`: useful multi-dimensional FRFT reference based on the Ozaktas chirp-convolution formulation.
- `nanaln/python_frft`: selected as the main reference because it implements a one-dimensional NumPy/SciPy fast FRFT, matching the TDI time-series data.
- `MStamatis/frft2d`: useful for two-dimensional data, but not used as the main method because this subtask transforms one-dimensional TDI time series.

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
4. Update this README with final result figures and conclusions.

## Expected Results

Subtask 1 should include:

- HDF5 structure inspection output.
- Cleaned time-series plot.
- Frequency-domain diagnostic plot.
- WDM time-frequency plot.
- FRFT result plot.

Subtask 2 should include:

- Baseline Example 4 result figures.
- Modified 5-day window result figures.
- Posterior corner plots.
- Parameter comparison table.
- Short analysis of how the 5-day window changes the inference result.

## Notes

This repository keeps third-party dependencies as external dependencies instead of copying their source code. Each third-party package follows its own license.
