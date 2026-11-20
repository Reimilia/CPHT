#!/usr/bin/env bash
#SBATCH --job-name=FJ_RL
#SBATCH --signal=USR1@600
#SBATCH --output=slurm-fj-%A_%a.out
set -euo pipefail
: "${BENCH_ROOT:?Set BENCH_ROOT to the extracted project}"
: "${PLAN:?Set PLAN to an absolute JSONL plan path}"
cd "$BENCH_ROOT"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
stage="${STAGE:-train}"
case "$stage" in
  prepare) command=prepare_fj.py ;;
  train) command=train_fj.py ;;
  eval) command=eval_fj.py ;;
  *) exit 2 ;;
esac
extra=()
if [[ "$stage" == train && "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ "$stage" == eval ]]; then
  extra+=(--cohorts "${COHORTS:-iid}" --clocks "${CLOCKS:-irregular,fixed}" --episodes "${EPISODES:-20}"
          --checkpoint "${CHECKPOINT_TAG:-final}" --split "${SPLIT:-test}"
          --video-episodes "${VIDEOS:-1}" --gl "${MUJOCO_GL:-egl}")
  if [[ "${OVERWRITE_VIDEOS:-0}" == 1 ]]; then extra+=(--overwrite-videos); fi
  if [[ -n "${VIDEO_SECONDS:-}" ]]; then extra+=(--video-seconds "$VIDEO_SECONDS"); fi
fi
default_device=cuda
if [[ "$stage" == eval ]]; then default_device=cpu; fi
srun python -u "$command" --plan "$PLAN" --index "${SLURM_ARRAY_TASK_ID:-0}" \
  --artifacts "${ARTIFACTS:-artifacts/fj}" --root "${RUN_ROOT:-runs}" \
  --device "${DEVICE:-$default_device}" "${extra[@]}"
