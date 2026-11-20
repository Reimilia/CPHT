#!/usr/bin/env bash
# One launcher for plan/calibration/train/F-J evaluation/report. Dry by default.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON:-python3}"
args=("${STAGE:-plan}" --profile "${PROFILE:-cases/cpht_comparison.json}"
      --run-root "${RUN_ROOT:-runs/cpht_comparison}"
      --artifacts "${ARTIFACTS:-artifacts/cpht_comparison}"
      --output "${REPORT:-reports/cpht_comparison.json}"
      --device "${DEVICE:-cpu}" --gl "${MUJOCO_GL:-egl}")
for pair in "METHODS:methods" "CASES:cases" "SEEDS:seeds" "INFORMATION_MODES:information" \
            "HORIZONS:horizons" "CONDITIONS:conditions" "CLOCKS:clocks" "TEST_CASES:test-cases" \
            "EPISODES:episodes" "EVAL_MAX_DECISIONS:eval-max-decisions"; do
    var="${pair%%:*}"
    flag="${pair#*:}"
    if [[ -n "${!var:-}" ]]; then args+=("--$flag" "${!var}"); fi
done
if [[ "${DRY_RUN:-1}" == 1 ]]; then args+=(--dry-run); fi
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ "${OVERWRITE:-0}" == 1 ]]; then args+=(--overwrite); fi
if [[ "${KEEP_GOING:-0}" == 1 ]]; then args+=(--keep-going); fi
exec "$python_bin" -m cpht_compare "${args[@]}"

