# Changes in v3.4

The source is the saved v3.3 archive with SHA256 7d70548adf81396dd2b7421141c73a9605da27e61604ac15235f54a16a0a7dae.

- Add Pure CPHT as a separate actor-free mechanical learning/control method.
- Construct the planar articulated mass matrix, bias, energy, and point Jacobians with vectorized Torch kinematics.
- Fit 32 positive mechanical parameters with a physical inverse-dynamics loss, episode holdout, and a positive prior.
- Fit four contact parameters from force/acceleration measurements. Predict contact through a finite-iteration unilateral Coulomb-cone QP.
- Add a causal point-force observer driven by the measured generalized environment port.
- Search a 19-parameter periodic reference potential through learned-model rollouts. Respect the measured first action hold and use a declared nominal clock for later predicted commands.
- Compute six bounded motor commands with a small active-set feedback QP. Retain floating-base underactuation and articulated coupling.
- Refit physical/contact models during training. Log model-energy recalibration jumps separately from actuator work and prediction integration error.
- Add saved-gait feedback-only deployment, receding-horizon feedback, headless recording, prediction diagnostics, and frozen F-to-J/terrain transfer.
- Extend the new preset with the existing noisy terrain probes for every selected method. Preserve the previous default observation/configuration identities.
- Add a reproducible one-second model-JSON feedback precheck. Include a zero-initial-amplitude option and a longer-horizon cycle preset.
- Add independent MuJoCo mechanics comparisons, canonical pH/power checks, perturbed-link learning, contact/observer checks, independent feedback-optimizer checks, causal hold checks, and actor-free recovery/rendering tests.

The supplied periodic initializer and short preview provide an actuation check. Learned-gait quality, full-course tracking, and environmental generalization require the server experiments. The CPHT value-space HJB/2PBVP solver remains a separate study.
