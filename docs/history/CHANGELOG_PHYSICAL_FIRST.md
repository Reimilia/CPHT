# Changes in v3.3

The input archive is the uploaded v3.2.1 bundle with SHA256 `f0d439707c92a8051f99402a79bd8d4eebfe2b897177d17f2de12da8d327226e`.

- Add the independent `physical_fj` pipeline: physics-rate acquisition, positive three-parameter calibration, episode holdout, fixed feature scaling, and measured-transition SAC/CT-SAC training.
- Add explicit shared state/force-port measurements. Acceleration labels remain calibration-only. Store contact-point Jacobians and coupled acceleration examples.
- Preserve raw acquisition before fitting. Record calibration failures and support cached refitting. Reject rank deficiency and held-out relative force error above 5%.
- Add method subsets, dry launches, CPU/CUDA selection, episode-boundary recovery, and final/intermediate policy checkpoints.
- Add checkpoint evaluation, frozen F-to-J and terrain transfer, paired seeds/clocks, headless MP4s, JSON, PNGs, and seed-level comparison.
- Include optional raw/physical cross controls to separate feature and critic-target effects. State the finite-path changes to the source-style CT-SAC target.
- Add legacy float64 inverse refinement, an absolute/relative gate, conditioning diagnostics, and failed-block rejection. Preserve calibration identities for numerical-only changes.
- Add `apply_finite_fix.py` for existing folders, `diagnose_finite_fj.py` for checkpoint inspection, and a reproducible numerical stress fixture.
- Add the physical-first guide, small dependency list, short smoke presets, and regression coverage. Preserve prior release notes and historical prechecks.

The physical-first method uses ordinary SAC value updates. Full physical-Hamiltonian discovery, online adaptation, new robot topologies, and the manuscript's HJB/2PBVP solver are outside this pilot. The checks include tiny controller updates; full experiment training remains unlaunched.
