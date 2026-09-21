#!/usr/bin/env bash
set -eo pipefail

repo_root="/mnt/e/TDCEnv/Repos/task5-lisa-taiji"
cache_root="/mnt/e/TDCEnv/Cache/task5-stage1"

mkdir -p \
  "${cache_root}/cupy" \
  "${cache_root}/cuda" \
  "${cache_root}/matplotlib" \
  "${cache_root}/numba" \
  "${cache_root}/torch" \
  "${cache_root}/xdg" \
  "${cache_root}/tmp" \
  "${cache_root}/pycache"

export TASK5_STAGE1_CACHE_DIR="${cache_root}"
export CUPY_CACHE_DIR="${cache_root}/cupy"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export TORCH_HOME="${cache_root}/torch"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TMPDIR="${cache_root}/tmp"
export TMP="${cache_root}/tmp"
export TEMP="${cache_root}/tmp"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate tri_env
cd "${repo_root}"

printf 'TASK5_STAGE1_CACHE_DIR=%s\n' "${TASK5_STAGE1_CACHE_DIR}"
printf 'CUPY_CACHE_DIR=%s\n' "${CUPY_CACHE_DIR}"
printf 'TMPDIR=%s\n' "${TMPDIR}"

python scripts/stage1_tc_closed_loop.py 2>&1 | tee \
  results/task5_subtask2_remediation/audit/stage1_tc_definition/stage1_closed_loop_run.log
