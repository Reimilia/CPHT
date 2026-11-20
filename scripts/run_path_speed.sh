#!/usr/bin/env bash
# Defaults only generate/inspect plans. DRY_RUN=0 explicitly executes a stage.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export CONFIG="${CONFIG:-cases/path_speed.json}"
export CONTROL_TASK=path_speed
export EXPERIMENTS="${EXPERIMENTS:-contact}"
export PLAN_DIR="${PLAN_DIR:-plans/path_speed}"
case "${CASE:-both}" in
  both) cases=(F J) ;;
  F|J) cases=("$CASE") ;;
  *) echo "CASE must be F, J, or both" >&2; exit 2 ;;
esac
for selected_case in "${cases[@]}"; do
  bash scripts/run_case.sh "$selected_case"
done
