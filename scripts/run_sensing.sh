#!/usr/bin/env bash
# Matched information ablations. Existing runners and presets remain available.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
IFS=',' read -r -a modes <<< "${INFORMATION_MODES:-state,state_belief,sensed_belief}"
base_runs="${RUN_ROOT:-runs/sensing}"
base_artifacts="${ARTIFACTS:-artifacts/sensing}"
export OPTIONS="${OPTIONS:-cases/sensing.json}"
export METHODS="${METHODS:-ct_sac,pure_cpht}"
for mode in "${modes[@]}"; do
  case "$mode" in state|state_belief|sensed_belief) ;; *) echo "Unknown information mode: $mode" >&2;exit 2 ;; esac
  INFORMATION="$mode" RUN_ROOT="$base_runs/$mode" ARTIFACTS="$base_artifacts/$mode" \
    bash "$project_dir/scripts/run_physical_first.sh"
done
