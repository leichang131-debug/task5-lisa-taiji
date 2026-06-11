#!/usr/bin/env bash
set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate tri_env

cd /mnt/e/TDCEnv/Repos/task5-lisa-taiji
mkdir -p results/task5_subtask2 .cache/matplotlib

ts="$(date +%Y%m%d_%H%M%S)"
log="results/task5_subtask2/notebook_02_full_run_${ts}.log"
pid_file="results/task5_subtask2/notebook_02_full_run.pid"
latest_file="results/task5_subtask2/notebook_02_full_run.latest"

echo "${log}" > "${latest_file}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/mnt/e/TDCEnv/Repos/task5-lisa-taiji/.cache/matplotlib"

nohup jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/02_taiji_mbhb_parameter_estimation.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  > "${log}" 2>&1 &

echo "$!" > "${pid_file}"
echo "started pid=$(cat "${pid_file}") log=${log}"
