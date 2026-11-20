#!/usr/bin/env bash
# Render a checkpoint in the existing v2 project. This script never starts training.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
: "${CHECKPOINT:?Set CHECKPOINT to final.pt or a policy_*.pt path}"
extra=()
if [[ -n "${EVALUATION_JSON:-}" ]]; then extra+=(--evaluation-json "$EVALUATION_JSON"); fi
if [[ -n "${CLOCK:-}" ]]; then extra+=(--clock "$CLOCK"); fi
if [[ -n "${MAX_VIDEO_SECONDS:-}" ]]; then extra+=(--max-video-seconds "$MAX_VIDEO_SECONDS"); fi
if [[ "${STOP_AFTER_VIDEO:-0}" == 1 ]]; then extra+=(--stop-after-video); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then extra+=(--dry-run); fi
if [[ "${OVERWRITE:-0}" == 1 ]]; then extra+=(--overwrite); fi
exec python render_fj.py --checkpoint "$CHECKPOINT" --out "${VIDEO_DIR:-videos/fj}" \
  --episodes "${EPISODES:-1}" --episode-offset "${EPISODE_OFFSET:-0}" \
  --gl "${MUJOCO_GL:-egl}" --device "${DEVICE:-cpu}" "${extra[@]}"
