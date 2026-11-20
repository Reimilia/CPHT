#!/usr/bin/env bash
# Independent, instrumented physical-calibration feature ablation. Dry by default.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
IFS=',' read -r -a methods <<< "${METHODS:-ct_sac,cpht_sac}"
IFS=',' read -r -a seeds <<< "${SEEDS:-0}"
for method in "${methods[@]}"; do
  case "$method" in ct_sac|cpht_sac|sac|ct_sac_ph|pure_cpht) ;; *) echo "Unknown physical-first method: $method" >&2;exit 2 ;; esac
done
case "${CASE:-both}" in both) cases=(F J) ;; F|J) cases=("$CASE") ;; *) exit 2 ;; esac
stage="${STAGE:-prepare}";terrain="${TERRAIN:-terrain}"
common=(--terrain "$terrain" --root "${RUN_ROOT:-runs/physical_first}" --artifacts "${ARTIFACTS:-artifacts/physical_first}")
if [[ -n "${OPTIONS:-}" ]]; then common+=(--options "$OPTIONS");fi
if [[ -n "${CONFIG:-}" ]]; then common+=(--config "$CONFIG");fi
if [[ -n "${INFORMATION:-}" ]]; then common+=(--information "$INFORMATION");fi
if [[ -n "${TOTAL_DECISIONS:-}" ]]; then common+=(--total-decisions "$TOTAL_DECISIONS");fi
if [[ -n "${CALIBRATION_DECISIONS:-}" ]]; then common+=(--calibration-decisions "$CALIBRATION_DECISIONS");fi
dry=();if [[ "${DRY_RUN:-1}" == 1 ]]; then dry+=(--dry-run);fi
for selected_case in "${cases[@]}"; do
  for seed in "${seeds[@]}"; do
    if [[ "$stage" == prepare ]]; then
      python prepare_physical_fj.py --case "$selected_case" --seed "$seed" "${common[@]}" "${dry[@]}"
    elif [[ "$stage" == train ]]; then
      extra=();if [[ "${RESUME:-0}" == 1 ]];then extra+=(--resume);fi
      for method in "${methods[@]}"; do
        python train_physical_fj.py --case "$selected_case" --seed "$seed" --method "$method" \
          --device "${DEVICE:-cuda}" "${common[@]}" "${dry[@]}" "${extra[@]}"
      done
    elif [[ "$stage" == eval ]]; then
      for method in "${methods[@]}"; do
        checkpoint="$(python -c 'from physical_fj.cli import parser,resolve; import sys; a=parser("",True).parse_args(sys.argv[1:]);print(resolve(a)[3]/"final.pt")' \
          --case "$selected_case" --seed "$seed" --method "$method" "${common[@]}")"
        extra=();if [[ -n "${TERRAIN_TEST:-}" ]];then extra+=(--terrain-test "$TERRAIN_TEST");fi
        if [[ -n "${CASE_TEST:-}" ]];then extra+=(--case-test "$CASE_TEST");fi
        if [[ -n "${SENSING_TEST:-}" ]];then extra+=(--sensing-test "$SENSING_TEST");fi
        if [[ "${FEEDBACK_ONLY:-0}" == 1 && "$method" == pure_cpht ]];then extra+=(--feedback-only);fi
        if [[ "${OVERWRITE_VIDEOS:-0}" == 1 ]];then extra+=(--overwrite-videos);fi
        python eval_physical_fj.py --checkpoint-file "$checkpoint" --cohorts "${COHORTS:-iid}" \
          --clocks "${CLOCKS:-fixed,irregular}" --episodes "${EPISODES:-5}" --video-episodes "${VIDEOS:-1}" \
          --device "${DEVICE:-cpu}" --gl "${MUJOCO_GL:-egl}" "${dry[@]}" "${extra[@]}"
      done
    else echo "STAGE must be prepare, train, or eval" >&2;exit 2;fi
  done
done
