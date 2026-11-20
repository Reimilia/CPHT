# Changes in v3

- Add F/contact and J/contact while retaining the four v2 subcases and published-source runners.
- Construct continuous piecewise-linear terrain from closed convex MJCF mesh wedges.
- Add spatial friction, synthetic contact compliance/damping, and factor-isolation profiles.
- Give ground priority 2 and verify the combined contact-pair coefficients.
- Add local relative terrain observations, a history-only ablation, and an explicit synthetic material-sensor study.
- Recompute dynamic observation sizes for replay, shared observer fitting, and checkpoint reconstruction.
- Add random terrain spawn locations and a declared J curriculum; expose full-range calibration levels and a no-curriculum configuration.
- Keep maps and parameter truth out of actor/replay features; record calibration exposure and contact audits separately.
- Add paired layout, terrain, friction, contact-response, and combined transfer cohorts with task-stratified aggregation.
- Add map sampling, profile plotting, EGL/OSMesa headless MP4 generation, camera tours, and frozen-checkpoint inspection.
- Copy simulator data before contact diagnostics so current force, point, and Jacobian agree without changing the rollout state.
- Count environmental samples and extra audit forward solves.
- Separate the simulator import path from upstream eager Torch imports without editing the pinned source archive.
- Add preview-only requirements, training/server launch instructions, 24 shared preparation jobs, and 192 contact training jobs.
- Move revised F/J configurations to schema 3 and record the current attached measurements-note hash.
- Check simulator/video dependencies at MuJoCo 3.17.0 and dm-control 1.0.50.
