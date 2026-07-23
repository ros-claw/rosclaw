#!/usr/bin/env bash
set -euo pipefail

repo_root="${ROSCLAW_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
lab_root="${ROSCLAW_ISAACLAB_ROOT:-$repo_root/.venv-isaaclab/src/IsaacLab}"
venv_root="${ROSCLAW_ISAACLAB_VENV:-$repo_root/.venv-isaaclab}"
gpu_list="${ROSCLAW_ISAAC_GPUS:-0,1,2,3}"
single_gpu="${ROSCLAW_ISAAC_SINGLE_GPU:-${gpu_list%%,*}}"
iterations="${ROSCLAW_ISAAC_ITERATIONS:-2}"
envs="${ROSCLAW_ISAAC_ENVS_PER_GPU:-4}"
timeout_sec="${ROSCLAW_ISAAC_TIMEOUT_SEC:-1800}"
master_port="${ROSCLAW_ISAAC_MASTER_PORT:-29504}"

if [[ ! "$gpu_list" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "ROSCLAW_ISAAC_GPUS must contain exactly four numeric CUDA indices" >&2
  exit 2
fi
IFS=, read -r -a gpu_ids <<<"$gpu_list"
if [[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "ROSCLAW_ISAAC_GPUS must contain four distinct CUDA indices" >&2
  exit 2
fi
if [[ ! "$single_gpu" =~ ^[0-9]+$ ]] || [[ ",${gpu_list}," != *",${single_gpu},"* ]]; then
  echo "ROSCLAW_ISAAC_SINGLE_GPU must be one of ROSCLAW_ISAAC_GPUS" >&2
  exit 2
fi
for value in "$iterations" "$envs" "$timeout_sec"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Isaac iteration, environment, and timeout values must be positive integers" >&2
    exit 2
  fi
done
if [[ ! "$master_port" =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
  echo "ROSCLAW_ISAAC_MASTER_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi

test -x "$venv_root/bin/python"
test -x "$lab_root/isaaclab.sh"
source "$venv_root/bin/activate"

gpu_count=4
CUDA_VISIBLE_DEVICES="$single_gpu" timeout "$timeout_sec" "$lab_root/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct \
  --num_envs "$envs" \
  presets=newton_mjwarp \
  --max_iterations "$iterations"

CUDA_VISIBLE_DEVICES="$gpu_list" timeout "$timeout_sec" "$lab_root/isaaclab.sh" -p \
  "$lab_root/scripts/reinforcement_learning/train_multigpu.py" \
  --num_gpus "$gpu_count" \
  --master_port "$master_port" \
  --tee 3 \
  --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct \
  --num_envs "$envs" \
  presets=newton_mjwarp \
  --max_iterations "$iterations"

echo "ROSCLAW_ISAACLAB_VERIFY_OK gpus=$gpu_list ranks=$gpu_count"
