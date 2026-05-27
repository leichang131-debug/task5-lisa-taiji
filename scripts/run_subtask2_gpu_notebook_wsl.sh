#!/usr/bin/env bash
set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate tri_env

cd /mnt/e/TDCEnv/Repos/task5-lisa-taiji
mkdir -p results/task5_subtask2

stamp="$(date +%Y%m%d_%H%M%S)"
log="results/task5_subtask2/notebook_02_gpu_run_${stamp}.log"

nohup jupyter nbconvert \
  --to notebook \
  --execute notebooks/02_taiji_mbhb_parameter_estimation.ipynb \
  --output 02_taiji_mbhb_parameter_estimation.executed.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=-1 \
  > "${log}" 2>&1 &

pid="$!"
echo "${pid}" > results/task5_subtask2/notebook_02_gpu_run.pid
echo "${log}" > results/task5_subtask2/notebook_02_gpu_run.latest
echo "${pid}"
echo "${log}"
