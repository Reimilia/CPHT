#!/usr/bin/env bash
# Set account, partition, GPU request, time, and array range at submission.
#SBATCH --job-name=tpht_train
#SBATCH --signal=USR1@600
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
: "${BENCH_ROOT:?Set BENCH_ROOT to the extracted benchmark directory}"
: "${PLAN:?Set PLAN to an absolute JSONL plan path}"
cd "$BENCH_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MUJOCO_GL=egl
export MPLBACKEND=Agg
resume_args=()
if [[ "${RESUME_MATCHED:-0}" == 1 ]]; then resume_args+=(--resume); fi
srun python -u train.py --plan "$PLAN" --root "${RUN_ROOT:-runs}" --device cuda "${resume_args[@]}"
