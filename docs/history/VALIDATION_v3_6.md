# Validation of v3.6

Date: 4 November 2026.

## Executed checks

- 16 unit/regression tests passed: seven new comparison tests, five existing mechanical checks, and four existing sensing-contract/observer checks.
- F and J shared calibration completed with 64 decisions per case/information mode.
- Four post-calibration policy decisions were executed for each method/case/mode. Learned variants made four selector updates; the value variant also made four critic updates.
- Two information modes were exercised: `state_belief` and `sensed_belief`.
- 120 eight-decision evaluation reports covered F/F, F-to-J, J/J, fixed/irregular holds, friction interventions, and applicable sensor degradation. All ran without execution errors. Strict pairing found 30 matched groups and no missing reports.
- A further 16 fixed-clock checks ran for 0.5 s each, covering all four variants in F and J with nominal and changed friction.
- Double-precision directional checks at one F and one J start produced finite gradients. Best relative errors were approximately `7.94e-9` and `2.65e-9` across the tested perturbations.

The gradient audit validates the derivative of the implemented surrogate at those starts. Contact branches and the finite contact solver remain part of that numerical map. True simulator-gradient fidelity is unverified.

## Observations

The final normalized diagnostics identified:

| Check | Observation | Interpretation |
| --- | --- | --- |
| Eight-decision sweep | 28 of 120 reports had no informative sliding evidence | Friction can remain unresolved under short/weakly excited contact histories |
| Eight-decision sweep | One report exceeded the configured one-step velocity-error threshold | Surrogate prediction needs independent checking |
| Half-second checks | All 16 had less than 1% path progress | Early progress diagnostic; it cannot distinguish ordinary startup from a persistent stall |
| Half-second checks | Two J reports exceeded the one-step velocity-error threshold | Contact/model/numerical mismatch can remain despite finite gradients |
| Half-second checks | No falls or execution exceptions during the checked interval | This interval is too short to establish stability |

In the half-second checks, progress was approximately 0-0.14 m and speed RMSE approximately 2.72-3.03 m/s for a 3 m/s target. These controllers received only four policy decisions, and imagination used a 20 ms horizon. They supply no trained ranking or claim about the proposed 0.2/0.4/0.6 s experiments.

The robot remained near the initial flat pad. These checks did not physically traverse the held-out hills or steps. The F/J transfer paths and intervention/sensor interfaces were exercised; terrain-generalization performance remains unmeasured.

All reported episodes were administratively budget-censored. Completion rates are therefore omitted. No full 100,000-decision run, full-course benchmark, independent-seed performance study, or GPU throughput test was performed.

## Included evidence

- `validation/cpht_smoke_report.json` and `.md`: all 120 short-report aggregates.
- `validation/cpht_halfsecond_report.json` and `.md`: the 16 half-second aggregates.
- `validation/episodes/`: representative nominal half-second traces for F/J and all four methods.
- `validation/gradient_check_F.json` and `gradient_check_J.json`.
- `validation/unit_tests.txt` and `micro_training.txt`.

The initial solver diagnostic mixed legacy conventions. The final reports use the dimensionless relative projected-step residual, and all affected evaluations were refreshed. The extra diagnostic computation leaves the physical transition and learned weights unchanged.

## Runtime and reproducibility

Checks used Python 3.12, Torch 2.9.1+cpu, MuJoCo 3.17.0, dm-control 1.0.50, Gymnasium 1.4.0, NumPy 2.5.4, and SciPy 1.18.1. CUDA execution is provided by the script interface but was not tested.

The retained dependency file pins the earlier benchmark's NumPy/SciPy versions. The checks above used the stated compatible installed versions. Wall times were measured during local checks and should not be interpreted as a controlled throughput benchmark.

To repeat the expanded short sweep:

```bash
python3 -m cpht_compare all \
  --profile cases/cpht_comparison_smoke.json \
  --information state_belief,sensed_belief \
  --clocks fixed,irregular --test-cases same,J \
  --conditions nominal,friction_change,sensor_shift \
  --run-root runs/cpht_smoke --artifacts artifacts/cpht_smoke \
  --output reports/cpht_smoke.json
```

To run the additional half-second evaluation, retain the smoke checkpoints:

```bash
python3 -m cpht_compare eval \
  --profile cases/cpht_comparison_smoke.json \
  --information state_belief --clocks fixed --test-cases same \
  --conditions nominal,friction_change --eval-max-decisions 50 \
  --run-root runs/cpht_smoke --artifacts artifacts/cpht_smoke
```

