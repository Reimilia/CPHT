# F/J courses aligned with the CT terrain branch

## Reference and scope

The physical reference is
[sebale16/ct-rl, branch cheetah-terrain](https://github.com/sebale16/ct-rl/blob/297ecbedb1b7510439f85927aa4b70d893f681e8/docs/cheetah_terrain.md),
pinned to commit `297ecbedb1b7510439f85927aa4b70d893f681e8`.
Its implementation is `environment/cheetah_terrain.py`, Git blob
`e95e50779d6389d2f1bc19b5f10f7fd8a518efec`.
This revision aligns course generation, collision geometry, friction precedence,
and the stock robot/contact parameters with that source.

The current policy measurements and sensors remain in use. The default task
still finishes a 20 m horizontal COM course while tracking 3 m/s before 12 s.
The source uses the stock cheetah running reward and a 10 s episode.
These objectives have different returns. Use the current completion and speed
metrics for the F/J path study.

## Physical course

The stock ground plane is removed. Fifty-two heightfield geoms cover
`[-5,203]` m along the forward axis. Each is 4 m long and 1.6 m wide.
Each heightfield has two rows and 81 columns at a 0.05 m spacing.
Adjacent patches share their boundary height sample.
The heightfield band is `[-1,1]` m, with a 0.5 m solid base below its lowest point.
The course has a finite end. It has no extra supporting plane or endpoint mesh extensions.

The course is level through x=3 m. Height features enter over the next 5 m
through the smoothstep $w(u)=u^2(3-2u)$, with $u$ clipped to $[0,1]$.
For hills, draw white noise on the global grid and apply a reflected Gaussian
filter with standard deviation

$$
\sigma_{\rm grid}=\frac{4\ {\rm m}}{2\pi(0.05\ {\rm m})}.
$$

Multiply the filtered signal by the start-pad blend. Rescale the result so its
largest absolute height over the full course is 0.3 m. Each geom receives its
slice of this global profile.

For friction patches, draw independently for each 4 m patch:

$$
\log\mu_i\sim{\rm Uniform}(\log0.1,\log1.0).
$$

Patches whose left edge is below 3 m keep the stock coefficient 0.4.
Friction then changes at patch boundaries. The five-metre smooth blend applies
to elevation. Torsional and rolling friction remain 0.1.
Ground priority is 1 and contact dimension is 3. Ground priority selects the
patch coefficient even when it is below the foot's own sliding coefficient.
Contact solver parameters inherit the stock cheetah XML defaults.

`hills` uses rolling elevation with uniform 0.4 friction.
`friction` uses level elevation with log-uniform patches.
`terrain` combines hills and patches. The source draws height noise before
friction. This random-number order is preserved, so a given seed has identical
heights in `hills` and `terrain`, while its friction draws can differ between
`friction` and `terrain`.

The optional `slopes` preset uses reflected piecewise-linear grades up to 0.25,
segment lengths from 2 to 8 m, and a height band of ±1 m.
The optional `steps` preset uses terraces with rises up to 0.1 m and segment
lengths from 1 to 4 m. Both include friction patches.
The implementation also preserves the source's grid interpolation at terrace changes.

The cheetah retains nine positions, nine velocities, six actuators, stock motor
gears, masses, inertias, joint damping, gravity, and integrator settings.
The matched presets apply no payload, wind, friction multiplier, material
response randomization, or difficulty curriculum.

## F, J, and the episode seed

F fixes terrain seed 1 across training, validation, test episodes, algorithms,
and clocks. This matches the source's `top_course1` physical course.
Robot reset and action sampling still use their declared episode seeds.

J draws a fresh course from the same physical law for each episode.
It uses the source's `numpy.random.RandomState` convention. For a given episode
seed, its course equals the source's first course draw after that seed is set.
The wrapper assigns explicit episode seeds so methods and clocks receive paired
courses. A source environment that continues one RNG stream through several
unseeded resets has its own sequence, which also includes pose-reset draws.

Policy evaluation uses seeds starting at 100000; validation starts at 50000.
The J `layout` cohort adds $2^{31}$ to its episode seed, modulo $2^{32}$, before
drawing the course. It uses the same parameter law with a separate seed stream.
J `iid` and `layout` test new layouts from the same distribution.
The optional `task` cohort changes target speed while
retaining the course law. F uses the `iid` cohort and separate validation seeds.

The earlier `terrain`, `friction`, `material`, `contact`, and robot-regime stress
cohorts remain features of the legacy mesh backend. They have explicit parameter
interventions. Choose `cases/path_speed_mesh.json` for that study.
The CT backend accepts `iid`, `layout`, and `task` at test time.
Select hills, friction, terrain, slopes, or steps through a physical preset.

The source reset settles a sampled joint pose on its start pad.
The path wrapper then aligns COM position with the course start and clears
velocity. Its calibration reset positions and speed strata remain as declared
in `PATH_SPEED_PROTOCOL.md`. Environment previews can request another start.
These reset/task additions are recorded separately from the matched physical model.

## Sample, inspect, train, and evaluate

Sample one F course and one J course, then inspect their physical geometry:

```bash
python sample_envs.py --case both --count 1 --config cases/path_speed_ct_terrain.json \
  --plot --out prechecks/ct_terrain/environments.jsonl
python preview_envs.py --manifest prechecks/ct_terrain/environments.jsonl \
  --limit 2 --seconds 2 --start-x 8 --out prechecks/ct_terrain/videos
python preview_envs.py --manifest prechecks/ct_terrain/environments.jsonl \
  --limit 2 --mode tour --start-x 3 --tour-end 25 --seconds 3 \
  --out prechecks/ct_terrain/videos
```

The preview controller is a fixed sine signal. Its video supplies a geometry
precheck. The checkpoint evaluator supplies trained-policy task metrics.

Preview a selected training subset:

```bash
TERRAIN=terrain CASE=both METHODS=ct_sac,ph_sac SEEDS=0 STAGE=train \
  bash scripts/run_terrain.sh
```

The command defaults to `DRY_RUN=1`. Set `TERRAIN=hills` or `friction` for the
other main variants. Preparation remains shared once per case and training seed.
To run on a compute server, retain the same settings and use the explicit stages:

```bash
export TERRAIN=terrain CASE=both METHODS=ct_sac,ph_sac SEEDS=0 DEVICE=cuda
STAGE=prepare DRY_RUN=0 bash scripts/run_terrain.sh
STAGE=train   DRY_RUN=0 bash scripts/run_terrain.sh
STAGE=eval EPISODES=5 VIDEOS=1 DRY_RUN=0 bash scripts/run_terrain.sh
```

Evaluation and checkpoint selection read the saved training plan for that variant.
The existing validation selection, paired clocks, JSON metrics, speed/progress
PNG plots, and headless MP4 pipeline apply to each variant.
Use `COHORTS=iid,layout` for J; F uses `COHORTS=iid`.
`README.md` gives the validation and aggregation commands.

Use a JSON override to customize supported physical scales before preparation.
With `run_terrain.sh`, `CONFIG=your_overrides.json` can supply neural budgets,
sensors, and physical scales. The selected variant's backend, profile, fixed
course seed, and elevation kind take precedence through `--base-config`.
The keys start with `terrain_`, such as `terrain_hill_height`,
`terrain_hill_wavelength`, `terrain_patch_length`, `terrain_resolution`,
`terrain_friction_min`, and `terrain_friction_max`.
Changing these values declares a new course distribution.
New physical settings create new calibration and run identities.
An existing checkpoint retains its saved physical backend when replayed.

## Reproduce the source check

Obtain the pinned upstream `environment/cheetah_terrain.py` file, then run:

```bash
python verify_terrain_match.py --upstream-file /path/to/ct-rl/environment/cheetah_terrain.py
python -m unittest discover -s tests -p test_ct_terrain.py -v
```

The verification script checks the upstream file's Git blob hash before import.
It compares ten courses: five variants at seeds 1 and 7. It checks exact course
arrays, compiled robot and ground arrays, contact parameters, and 997 vertical
surface rays per course. `prechecks/ct_terrain/upstream_match.json` records the
executed comparison. The independent numerical fixtures are in `tests/fixtures/`.
The raw upstream module is an external verification input.

The match was checked with the same installed dm-control and MuJoCo stack on
both sides. The adapter writes heightfield data after compilation, preserving
the source's normalization. The current collector creates an environment per
episode; it supplies no claim about matching the source's reset throughput.
No costly training was launched for this revision.
