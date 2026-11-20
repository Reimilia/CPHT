#!/usr/bin/env bash
# Environment previews run only when DRY_RUN=0. Training remains an explicit stage.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
case_name="${CASE:-J}"
stage="${STAGE:-sample}"
manifest="${MANIFEST:-prechecks/contact/${case_name}_environments.jsonl}"
config_args=()
if [[ -n "${CONFIG:-}" ]]; then config_args+=(--config "$CONFIG"); fi
dry_args=()
if [[ "${DRY_RUN:-1}" == 1 ]]; then dry_args+=(--dry-run); fi
case "$stage" in
  sample)
    python sample_envs.py --case "$case_name" --count "${COUNT:-3}" --cohort "${COHORT:-iid}" \
      --progress "${PROGRESS:-1}" --out "$manifest" --plot "${config_args[@]}" "${dry_args[@]}"
    ;;
  preview)
    python preview_envs.py --manifest "$manifest" --out "${VIDEO_DIR:-prechecks/contact/videos}" \
      --mode "${MODE:-rollout}" --policy "${POLICY:-sine}" --seconds "${SECONDS_TO_RENDER:-3}" \
      --start-x "${START_X:-4}" --gl "${MUJOCO_GL:-egl}" --limit "${COUNT:-3}" "${dry_args[@]}"
    ;;
  prepare|train|eval)
    export EXPERIMENTS=contact
    export STAGE="$stage"
    exec bash scripts/run_case.sh "$case_name"
    ;;
  *) echo "STAGE must be sample, preview, prepare, train, or eval" >&2; exit 2 ;;
esac
