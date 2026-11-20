# Validation of v3.2.1: preparation goal coverage

Checks ran on 27 October 2026. The full suite passed: **78 tests, zero failures,
zero skips**, in 57.999 s. The log is `prechecks/calibration/unit_tests.txt`.
No full calibration or experiment training was launched.

## Error and correction

The earlier preparation guard required goal examples in both episode
partitions. Random acquisition and an every-fifth split did not ensure that
coverage. New regression fixtures reproduce the case where both observed goal
episodes went to training and validation received zero goals.

The corrected collector uses bounded measured goal probes inside the declared
interaction budget. The corrected split groups whole episodes by target task
and observed terminal outcome. It requires goal coverage per task in both
partitions. Failed probes remain failures; the guard remains active.

## Executed preparation checks

- Data-only F/J acquisitions used 128 decisions per case. Each supplied three
  measured goal episodes to training and one to validation. No neural updates
  were performed by `check_calibration_fj.py`.
- Simulator regression tests repeat the short acquisition on both CT terrain
  and the earlier mesh backend. Goal records have actual measured rewards and
  true termination flags. Decision counters match the fixed budget.
- A two-target J check covers target speeds 2.7 and 3.3 m/s separately.
- Episode-split tests verify coverage, disjoint episodes, unchanged record
  membership, deterministic splitting, singleton handling, and missing-label
  rejection.
- Bounded failed-probe tests verify that failed physical attempts produce zero
  successes and stop early. Missing coverage writes acquisition/diagnostics and
  stops before any observer or operator fit.
- Tiny preparation/retry tests simulate an interruption after acquisition, then
  complete fitting from the saved `acquired.pt` while forbidding reacquisition.
  The resulting `prepared.pt` files load successfully.
- The exact `scripts/run_path_speed.sh` launcher completed `STAGE=prepare` for
  both F and J under a tiny configuration: 128 decisions, one observer update,
  and one update per operator member. F has one member; J has three. There were
  zero policy updates in these launcher runs. The results are summarized in
  `prechecks/calibration/cli_prepare_smoke.json`.
- The installer test applies the fix to an existing folder named
  `TPHT_CT_RL_Benchmark_v2` and verifies the source-file backup.
- All 1032 supplied plan entries were regenerated and checked for configuration,
  calibration identity, and run identity. These remain unlaunched job descriptions.
- Python compilation, shell syntax, and dry preparation launches passed.

The short acquisition reports are in `prechecks/calibration/`. Their output
supplies a preparation check, rather than a trained control-performance result.
The full 30000-decision acquisition and 2000/5000 observer/operator fitting
budgets remain server work.

## Runtime and retained checks

The checks used Python 3.12, Torch 2.9.1+cpu, NumPy 2.3.5, SciPy 1.17.0,
MuJoCo 3.17.0, dm-control 1.0.50, and EGL. CUDA execution was not tested here.

The physical terrain implementation and actor/sensor interfaces are unchanged
from v3.2. The passing regression suite includes the independent terrain
fixtures, surface/friction tests, F/J evaluation under both clocks, video
invariance checks, and earlier solver/adapter checks.
`VALIDATION_v3_2.md` retains the ten-course upstream model comparison and eight
headless environment previews. Earlier reports remain in their versioned files.

Full training quality, exploration completeness, physical safety, and policy
optimality remain outside these checks. Calibration admission confirms measured
coverage of the stated boundary classes and disjoint data partitions.
