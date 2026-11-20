#!/usr/bin/env bash
# Run the CPHT experiment at one scale.
#
#   ./run_experiments.sh [SCALE] [STAGE]
#
#   SCALE  smoke | pilot | full          (default: smoke)
#   STAGE  plan | prepare | gradcheck | train | eval | report | verify | all
#                                        (default: all; full scale defaults to plan)
#
# Profiles live in experiments/<scale>.json. Outputs go to runs/<scale>,
# artifacts/<scale> and reports/<scale>.json. Rerunning a finished stage
# reuses its results; a partly trained run continues with RESUME=1.
#
# Optional environment variables narrow or adjust a run without editing
# the profile:
#   METHODS=cem,shooting,selector,selector_value,ct_sac   CASES=F,J   SEEDS=0,1,2
#   INFORMATION_MODES=state,state_belief,sensed_belief     HORIZONS=0.2,0.4
#   CONDITIONS=nominal,layout,friction_change,sensor_shift,steps
#   CLOCKS=fixed,irregular   TEST_CASES=same,F,J   EPISODES=5   EVAL_MAX_DECISIONS=N
#   DEVICE=cpu|cuda   RESUME=1   OVERWRITE=1   KEEP_GOING=1   DISCLOSE_HOLDS=1
#   PROFILE=path.json   SENSOR_ROOT=/path/to/sensor (verify: MuJoCo reference audit)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

scale="${1:-${SCALE:-smoke}}"
case "$scale" in
  smoke|pilot|full) ;;
  *) echo "Unknown scale '$scale'. Use smoke, pilot or full." >&2; exit 2 ;;
esac
default_stage=all
[[ "$scale" == full ]] && default_stage=plan
stage="${2:-${STAGE:-$default_stage}}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
args=("$stage" --profile "${PROFILE:-experiments/$scale.json}"
      --run-root "runs/$scale" --artifacts "artifacts/$scale"
      --output "reports/$scale.json" --device "${DEVICE:-cpu}" --gl "$MUJOCO_GL")
for pair in METHODS:methods CASES:cases SEEDS:seeds INFORMATION_MODES:information \
            HORIZONS:horizons CONDITIONS:conditions CLOCKS:clocks TEST_CASES:test-cases \
            EPISODES:episodes EVAL_MAX_DECISIONS:eval-max-decisions SENSOR_ROOT:sensor-root; do
  var="${pair%%:*}"
  if [[ -n "${!var:-}" ]]; then args+=("--${pair#*:}" "${!var}"); fi
done
[[ "${RESUME:-0}" == 1 ]] && args+=(--resume)
[[ "${OVERWRITE:-0}" == 1 ]] && args+=(--overwrite)
[[ "${KEEP_GOING:-0}" == 1 ]] && args+=(--keep-going)
[[ "${DISCLOSE_HOLDS:-0}" == 1 ]] && args+=(--disclose-holds)

exec "${PYTHON:-python3}" -m cpht_compare "${args[@]}"
