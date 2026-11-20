# Changes in v3.2

- Add CT terrain heightfields with the source's 4 m patches, 0.05 m grid, start pad, rolling hills, log-uniform friction, and ground contact priority 1.
- Add hills-only, friction-only, combined, slopes, and steps presets. F fixes course seed 1; J samples a fresh full-difficulty course per episode.
- Keep the current observations, finite path/speed task, costs, algorithms, clocks, and evaluation outputs. Retain the earlier mesh/material environments through explicit presets.
- Add a terrain launcher with separate saved plans per variant, protected physical presets when merging custom neural JSON, and the existing algorithm-subset controls.
- Record the pinned terrain source in configuration/run identity, source provenance, and backend audits.
- Add independent numerical course fixtures and a hash-checked source-comparison script.
- Add F/J headless prechecks, physical surface/friction tests, and checkpoint evaluation tests under both clocks.
- Permit fixed F contact maps on the validation split, keeping the same map and separate rollout seeds.
- Reject comparison plots with different physical/preparation specifications or interaction budgets.
- Preserve previous README and validation reports, and add TERRAIN_ALIGNMENT.md.
