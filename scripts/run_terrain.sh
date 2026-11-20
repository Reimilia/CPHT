#!/usr/bin/env bash
# Select one matched physical course family. All stages default to dry run.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
terrain="${TERRAIN:-terrain}"
case "$terrain" in
  hills|friction|terrain|slopes|steps) ;;
  *) echo "TERRAIN must be hills, friction, terrain, slopes, or steps" >&2; exit 2 ;;
esac
export BASE_CONFIG="cases/path_speed_ct_${terrain}.json"
export CONFIG="${CONFIG:-$BASE_CONFIG}"
export PLAN_DIR="${PLAN_DIR:-plans/ct_terrain/$terrain}"
bash scripts/run_path_speed.sh
