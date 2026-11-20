# Validation of the v3 contact extension

Checks were run on 25 October 2026. No observer, operator, or policy fit was launched.
The four MP4s are short environment prechecks with a fixed sine controller or a camera tour.

## Executed checks

- 44 unit tests discovered: **40 passed, 4 skipped**. The four skipped tests require Torch.
- Optional MuJoCo tests passed: low ground friction remains effective; back/front foot Jacobian support follows the articulated tree; sensor size and actuator count agree; force auditing leaves trajectory and warm-start arrays unchanged.
- Nine profile/cohort simulator checks passed with finite observations and matching effective friction, contact time, and damping. See `prechecks/contact/backend_matrix.json`.
- Six F/J subcases were constructed and advanced for two zero-action decisions in the backend check.
- F and J each produced a two-second, 50-frame, 800-by-480 rollout MP4 through EGL. Each also produced a 50-frame camera-tour MP4.
- Rendered first frames, middle frames, and terrain profile figures were visually inspected.
- Rollout friction/contact-response audit errors were zero for all checked pairs.
- F rollout maximum penetration was about 0.00098 m. J rollout maximum penetration was about 0.01058 m. These values describe the supplied synthetic soft-contact examples.
- Map tests verify fixed F maps, disjoint J layout banks, paired single-factor interventions, continuous wedge tops with outward faces, and permitted sensor fields.
- Synthetic aggregation fixtures verify that contact metrics average known task strata before independent training seeds.
- All 48 original-family preparation, 384 original-family training, 24 contact preparation, and 192 contact training plan entries passed configuration and identity checks.
- Python syntax checks, shell syntax checks, dependency-free dry runs, and contact wrapper previews passed.

The simulator/video stack was Python 3.12, NumPy 2.3.5, MuJoCo 3.17.0, dm-control 1.0.50,
Gymnasium 1.4.0, imageio 2.38.1, and imageio-ffmpeg 0.6.0.
System EGL rendered without an attached display. Torch was absent from this preparation environment.
The source model emits a `settotalmass` deprecation message on this simulator version; its compiled robot mass remains the source 14 kg before an optional payload.

## Checks for the compute server

Run the full unit suite with Torch, then one small preparation/train/evaluation smoke budget before submitting arrays.
Verify checkpoint recovery on the selected Torch/CUDA stack.
The four skipped neural checks have no substituted result.
This revision includes no claim of training success, policy improvement, calibrated material physics, exact belief closure, physical safety, or a global learning theorem.

Use a physics-step refinement study before reporting trajectory, slip, or energy accuracy.
The audit uses endpoint contact re-solves on copied data and endpoint force/velocity quadrature.
Its extra forward solves are counted separately and contribute to wall-clock simulation cost.
Default neural fitting remains empirical admission; strict numerical admission is a separate declared study.

Historical preparation checks remain in `VALIDATION_v2.md` and `VALIDATION_v1.md`.
