# v2 F/J revision

- Added four measured Cheetah experiments: F/native, F/position, J/static, and J/wind.
- Added common observer/operator preparation, public raw-history replay, and episode-held-out validation.
- Added family-conditioned positive operator fitting with terminal-value probes and explicit work accounting.
- Added true free twin-critic residual recovery and the corresponding cotangent calculation.
- Replaced average irregular-h entropy in the new cases with explicit per-duration conventions.
- Added CT-SAC-I and CT-TD3 source-equation adapters on the common F/J interface.
- Added model-backed, Euler, pH, and direct-solve comparisons.
- Added empirical and strict returned-critic admission modes with visible fitting/storage debts.
- Added episode-boundary recovery, frozen-policy holdout cohorts, paired-seed aggregation, and F/J launchers.
- Preserved the earlier four-robot suite and its documentation as the v1 reference.

The package contains no new training result. Validation scope is recorded in `VALIDATION.md`.
