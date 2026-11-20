#!/usr/bin/env bash
# Actor-free learned pH mechanics, local contact prediction, and feedback control.
set -euo pipefail
export METHODS="${METHODS:-pure_cpht}"
export OPTIONS="${OPTIONS:-cases/pure_cpht.json}"
export DEVICE="${DEVICE:-cpu}"
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/run_physical_first.sh"
