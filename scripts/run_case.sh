#!/usr/bin/env bash
# DRY_RUN=1 prints configurations. Set DRY_RUN=0 to execute on a compute server.
set -euo pipefail
case_name="${1:?Pass F or J}"
case "$case_name" in F|J) ;; *) exit 2 ;; esac
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
stage="${STAGE:-train}"
algorithm_args=(--algorithm-set "${ALGORITHM_SET:-core}")
if [[ -n "${METHODS+x}" ]]; then algorithm_args+=(--methods "$METHODS"); fi
seeds="${SEEDS:-0}"
device="${DEVICE:-cuda}"
experiments="${EXPERIMENTS:-}"
plan_dir="${PLAN_DIR:-plans}"
artifacts="${ARTIFACTS:-artifacts/fj}"
run_root="${RUN_ROOT:-runs}"
mkdir -p "$plan_dir"
case_args=()
if [[ -n "$experiments" ]]; then case_args+=(--experiments "$experiments"); fi
config_args=()
if [[ -n "${CONFIG:-}" ]]; then config_args+=(--config "$CONFIG"); fi
if [[ -n "${BASE_CONFIG:-}" ]]; then config_args+=(--base-config "$BASE_CONFIG"); fi
if [[ -n "${PATH_LENGTH:-}" ]]; then config_args+=(--path-length "$PATH_LENGTH"); fi
if [[ -n "${TARGET_SPEED:-}" ]]; then config_args+=(--target-speed "$TARGET_SPEED"); fi
if [[ -n "${DEADLINE:-}" ]]; then config_args+=(--deadline "$DEADLINE"); fi
if [[ -n "${CONTROL_TASK:-}" ]]; then config_args+=(--control-task "$CONTROL_TASK"); fi
saved_stage_plan=""
if [[ ( "$stage" == eval || "$stage" == select ) && -f "$plan_dir/${case_name}_train.jsonl" ]]; then
  filter_args=()
  if [[ -n "${METHODS+x}" ]]; then filter_args+=(--methods "$METHODS"); fi
  if [[ -n "${SEEDS+x}" ]]; then filter_args+=(--seeds "$SEEDS"); fi
  saved_stage_plan="$plan_dir/${case_name}_${stage}.jsonl"
  python make_fj_plan.py --case "$case_name" --kind train --from-plan "$plan_dir/${case_name}_train.jsonl" \
    "${filter_args[@]}" --out "$saved_stage_plan"
else
  python make_fj_plan.py --case "$case_name" --kind prepare --seeds "$seeds" \
    "${case_args[@]}" "${config_args[@]}" --out "$plan_dir/${case_name}_prepare.jsonl"
  python make_fj_plan.py --case "$case_name" --kind train --seeds "$seeds" "${algorithm_args[@]}" \
    "${case_args[@]}" "${config_args[@]}" --out "$plan_dir/${case_name}_train.jsonl"
fi
if [[ "$stage" == prepare ]]; then
  entry=prepare_fj.py; plan="$plan_dir/${case_name}_prepare.jsonl"
elif [[ "$stage" == train ]]; then
  entry=train_fj.py; plan="$plan_dir/${case_name}_train.jsonl"
elif [[ "$stage" == eval ]]; then
  entry=eval_fj.py; plan="${saved_stage_plan:-$plan_dir/${case_name}_train.jsonl}"; device=cpu
elif [[ "$stage" == select ]]; then
  entry=select_fj_checkpoint.py; plan="${saved_stage_plan:-$plan_dir/${case_name}_train.jsonl}"; device=cpu
else
  echo "STAGE must be prepare, train, eval, or select" >&2; exit 2
fi
job_count="$(python -c 'import sys;print(sum(bool(s.strip()) for s in open(sys.argv[1])))' "$plan")"
extra=()
if [[ "${DRY_RUN:-1}" == 1 ]]; then extra+=(--dry-run); fi
if [[ "$stage" == train && "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ "$stage" == eval ]]; then
  extra+=(--cohorts "${COHORTS:-iid}" --clocks "${CLOCKS:-irregular,fixed}" --episodes "${EPISODES:-20}"
          --checkpoint "${CHECKPOINT_TAG:-final}" --split "${SPLIT:-test}" --video-episodes "${VIDEOS:-1}" --gl "${MUJOCO_GL:-egl}")
fi
if [[ "$stage" == eval && "${OVERWRITE_VIDEOS:-0}" == 1 ]]; then extra+=(--overwrite-videos); fi
if [[ "$stage" == eval && -n "${VIDEO_SECONDS:-}" ]]; then extra+=(--video-seconds "$VIDEO_SECONDS"); fi
if [[ "$stage" == select ]]; then extra+=(--clocks "${CLOCKS:-irregular,fixed}" --episodes "${EPISODES:-20}"); fi
for ((index=0; index<job_count; index++)); do
  python "$entry" --plan "$plan" --index "$index" --artifacts "$artifacts" \
    --root "$run_root" --device "$device" "${extra[@]}"
done
