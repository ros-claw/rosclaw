#!/usr/bin/env bash
set -euo pipefail

repo_root="${ROSCLAW_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
lab_root="${ROSCLAW_ISAACLAB_ROOT:-$repo_root/.venv-isaaclab/src/IsaacLab}"
venv_root="${ROSCLAW_ISAACLAB_VENV:-$repo_root/.venv-isaaclab}"
gpu_list="${ROSCLAW_ISAAC_GPUS:-0,1,2,3}"
single_gpu="${ROSCLAW_ISAAC_SINGLE_GPU:-${gpu_list%%,*}}"
iterations="${ROSCLAW_ISAAC_ITERATIONS:-2}"
envs="${ROSCLAW_ISAAC_ENVS_PER_GPU:-4}"

test -x "$venv_root/bin/python"
test -x "$lab_root/isaaclab.sh"
source "$venv_root/bin/activate"

gpu_count=$(awk -F, '{print NF}' <<<"$gpu_list")
CUDA_VISIBLE_DEVICES="$single_gpu" "$lab_root/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct \
  --num_envs "$envs" \
  presets=newton_mjwarp \
  --max_iterations "$iterations"

CUDA_VISIBLE_DEVICES="$gpu_list" "$lab_root/isaaclab.sh" -p \
  "$lab_root/scripts/reinforcement_learning/train_multigpu.py" \
  --num_gpus "$gpu_count" \
  --master_port "${ROSCLAW_ISAAC_MASTER_PORT:-29504}" \
  --tee 3 \
  --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct \
  --num_envs "$envs" \
  presets=newton_mjwarp \
  --max_iterations "$iterations"

echo "ROSCLAW_ISAACLAB_VERIFY_OK gpus=$gpu_list ranks=$gpu_count"
