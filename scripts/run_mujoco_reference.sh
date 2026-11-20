#!/usr/bin/env bash
# No training. Set SENSOR_ROOT to the extracted sensor/ directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SENSOR_ROOT:?Set SENSOR_ROOT to the extracted sensor directory containing F and J}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-outputs/mujoco_reference}"
MODE="${MODE:-compare}"
case "$MODE" in
  compare)
    "$PYTHON" -m mujoco_reference compare --sensor-root "$SENSOR_ROOT" --output "$OUT" "$@"
    ;;
  sample)
    "$PYTHON" -m mujoco_reference sample --sensor-root "$SENSOR_ROOT" --output "$OUT/samples" "$@"
    ;;
  *) echo 'MODE must be compare or sample' >&2; exit 2 ;;
esac
