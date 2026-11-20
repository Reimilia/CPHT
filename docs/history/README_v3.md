# MuJoCo benchmark v3: contact environments for F and J

This revision adds `--experiment contact` to both measured-data cases.
Read [CONTACT_BENCHMARK.md](../CONTACT_BENCHMARK.md) for the formulation, measurements, curriculum, and evaluation protocol.
The four v2 experiments and their methods remain available; [README_v2.md](README_v2.md) describes them.
The unchanged published runners remain under [README_v1.md](README_v1.md).

- **F/contact:** one fixed elevation and material map across episodes, with randomized robot resets.
- **J/contact:** sampled maps, a declared difficulty curriculum, and separate layout, elevation, friction, and contact-response transfer tests.
- Every method retains six motor commands and receives the same declared sensor.
- Material presets specify sliding friction and MuJoCo contact time/damping parameters. Their values are synthetic.

## Preview without training

Use Python 3.11 or 3.12. The preview environment requires no Torch installation:

```bash
python -m venv .venv-preview
source .venv-preview/bin/activate
python -m pip install -r requirements-preview.txt

python sample_envs.py --case both --count 1 --plot \
  --out prechecks/contact/demo_environments.jsonl
MUJOCO_GL=egl python preview_envs.py \
  --manifest prechecks/contact/demo_environments.jsonl --limit 2 --seconds 3
MUJOCO_GL=egl python preview_envs.py \
  --manifest prechecks/contact/demo_environments.jsonl --limit 2 --seconds 3 --mode tour
```

The output contains MP4 videos, profile figures, and per-step contact audit files.
The supplied `prechecks/contact/videos/` folder contains two-second EGL examples for F and J.
Their sine controller has no fitted parameters. A camera tour moves only the camera.
Both are environment checks; they supply no trained-policy result.

Set `--gl osmesa` for CPU rendering when OSMesa is installed.
Select the backend before importing MuJoCo. A render failure reports the error and stops.
Use `--checkpoint PATH --device cpu` to preview a frozen matching policy later; that option requires Torch.

## Choose the physical factors and measurements

```bash
# Elevation with uniform contact parameters.
python sample_envs.py --case J --config cases/contact_elevation.json --plot
# Flat surfaces with spatially varying friction.
python sample_envs.py --case J --config cases/contact_friction.json --plot
# Flat surfaces with varying friction and contact response.
python sample_envs.py --case J --config cases/contact_material.json --plot
# Independent held-out friction intervention on a mixed map.
python sample_envs.py --case J --cohort friction --plot
```

The default `mixed` profile combines hills, friction patches, and contact-response patches.
The default sensor gives local relative heights and slopes. Friction and contact-response values stay hidden.
`cases/contact_sensor_history.json` removes the environmental sensor.
`cases/contact_sensor_material.json` adds a declared synthetic noisy sensor for local friction, contact time, and damping.
Apply the same file to every compared method and to preparation, training, and evaluation.
The material sensor defines a separate information experiment.

## Prepare, train, and evaluate on the compute server

Use the training environment from `requirements.txt`, with a Torch wheel appropriate for the server.
These stage commands launch work when `DRY_RUN=0`:

```bash
# Run each preparation once, then share it across methods.
CASE=F STAGE=prepare DRY_RUN=0 DEVICE=cuda bash scripts/run_contact.sh
CASE=J STAGE=prepare DRY_RUN=0 DEVICE=cuda bash scripts/run_contact.sh

CASE=F STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_contact.sh
CASE=J STAGE=train DRY_RUN=0 DEVICE=cuda bash scripts/run_contact.sh

CASE=F STAGE=eval DRY_RUN=0 DEVICE=cpu COHORTS=iid bash scripts/run_contact.sh
CASE=J STAGE=eval DRY_RUN=0 DEVICE=cpu \
  COHORTS=iid,layout,terrain,friction,material,contact,task,joint bash scripts/run_contact.sh
```

The contact wrapper defaults to seed 0 and eight methods. Set `SEEDS=0,1,2`, `METHODS`, or `CONFIG` as needed.
Every wrapper defaults to `DRY_RUN=1`. The following commands only print planned work:

```bash
CASE=J STAGE=train bash scripts/run_contact.sh
python train_fj.py --case J --experiment contact --method ph_sac --dry-run
```

The supplied full contact plans cover twelve seeds, eight methods, and both cases:

```bash
python prepare_fj.py --plan plans/contact_prepare.jsonl --index 0 --device cuda
python train_fj.py --plan plans/contact_train.jsonl --index 0 --device cuda
python eval_fj.py --plan plans/contact_train.jsonl --index 0 --device cpu
```

There are 24 preparation jobs and 192 training jobs. Reuse `scripts/slurm_fj.sh` with those plans.
A single job keeps the v2 budget of 1,000,000 decisions, including 30,000 calibration decisions.
No training, observer fit, operator fit, or policy evaluation was launched while preparing this revision.

Schema 3 gives the revised configurations new calibration and run identities.
Regenerate plans with the supplied code. Keep an existing v2 checkpoint with its v2 bundle.
The released source archive remains hash pinned. The checked simulator stack is MuJoCo 3.17.0 and dm-control 1.0.50.

[VALIDATION.md](VALIDATION.md) records the executed checks and remaining server checks.
