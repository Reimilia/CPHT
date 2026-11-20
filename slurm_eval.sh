#!/usr/bin/env bash
# Evaluation normally needs CPU resources only. Set the array range at submission.
#SBATCH --job-name=tpht_eval
#SBATCH --output=slurm-eval-%A_%a.out
set -euo pipefail
: "${BENCH_ROOT:?Set BENCH_ROOT to the extracted benchmark directory}"
: "${PLAN:?Set PLAN to an absolute JSONL plan path}"
cd "$BENCH_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MUJOCO_GL=egl
export MPLBACKEND=Agg
srun python -u eval.py --plan "$PLAN" --root "${RUN_ROOT:-runs}" --device cpu \
  --protocol "${EVAL_PROTOCOL:-common}" --contexts "${CONTEXTS:-nominal}" --episodes "${EPISODES:-20}"
