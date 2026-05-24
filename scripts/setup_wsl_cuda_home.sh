#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_PREFIX="/opt/miniforge3/envs/tri_env"
CUDA_HOME_COMPAT="${CONDA_ENV_PREFIX}/cuda-home"

rm -rf "${CUDA_HOME_COMPAT}"
mkdir -p "${CUDA_HOME_COMPAT}/bin"

ln -s "${CONDA_ENV_PREFIX}/bin/nvcc" "${CUDA_HOME_COMPAT}/bin/nvcc"
ln -s "${CONDA_ENV_PREFIX}/targets/x86_64-linux/include" "${CUDA_HOME_COMPAT}/include"
ln -s "${CONDA_ENV_PREFIX}/lib" "${CUDA_HOME_COMPAT}/lib64"

echo "CUDAHOME compatibility directory:"
ls -l "${CUDA_HOME_COMPAT}" "${CUDA_HOME_COMPAT}/bin"
