# Headless video patch for existing F/J checkpoints

The v2 `eval_fj.py` writes numerical evaluation JSON files.
This additive patch supplies `render_fj.py` and a shell launcher.
It uses the existing saved actor and observation encoder and performs no learning update.
It works with schema-2 checkpoints from `TPHT_CT_RL_MuJoCo_Benchmark_v2_FJ.zip`.
Keep the original project, configuration, and simulator environment for this trained run.

## 1. Install the patch into your existing project

Extract this patch ZIP **inside the original `TPHT_CT_RL_Benchmark` directory**, alongside `train_fj.py`, `eval_fj.py`, `fjbench/`, and `bench/`.
The ZIP has no enclosing project directory. Its files add the renderer and instructions.
It changes no training module, configuration, existing metric JSON, or checkpoint.

Activate the same Python environment used for the evaluation, then add the video packages:

```bash
python -m pip install -r requirements-video.txt
```

Retain the installed Torch, MuJoCo, and dm-control versions.
A working headless OpenGL backend is also required.
EGL is suitable for a headless server; `--gl osmesa` uses the system OSMesa software renderer.
Set `MUJOCO_EGL_DEVICE_ID` externally when a server requires selection of a particular EGL device.
Policy inference defaults to CPU. The graphics backend and policy device are separate choices.

## 2. Render the checkpoint evaluated in your uploaded JSON

Your two records describe `F/native`, method `ct_sac`, training seed 0, and the same `final.pt` checkpoint.
The checkpoint SHA-256 in both records is

```text
f1d071878de04e7be3667c233134b3a9ef45604770f2efa4c43572a9dcbecc3e
```

With the default v2 run root, the expected checkpoint path is:

```text
runs/FJ/F/native/ct_sac/seed0_1951ffde74e1/final.pt
```

Use the actual path if your run directory was moved or you selected another `--root`.
The evaluation JSON files are normally inside that run's `evaluations/` directory.

```bash
RUN_DIR=runs/FJ/F/native/ct_sac/seed0_1951ffde74e1

# Reproduce and film the first fixed-hold evaluation episode.
python render_fj.py \
  --checkpoint "$RUN_DIR/final.pt" \
  --evaluation-json "$RUN_DIR/evaluations/final_test_iid_fixed_n20.json" \
  --episodes 1 --out videos/F_native_ct_sac/fixed --gl egl --device cpu

# Reproduce and film the corresponding irregular-hold evaluation episode.
python render_fj.py \
  --checkpoint "$RUN_DIR/final.pt" \
  --evaluation-json "$RUN_DIR/evaluations/final_test_iid_irregular_n20.json" \
  --episodes 1 --out videos/F_native_ct_sac/irregular --gl egl --device cpu
```

The JSON supplies the clock, cohort, split, original episode index, and reset seed.
Do not add a conflicting `--clock` or seed override.
The renderer verifies the checkpoint hash, method, configuration, and calibration identity before recording.
It reports model/backend differences and stops on those differences by default.
`--allow-backend-mismatch` permits a separately labelled diagnostic replay when the original simulator stack is unavailable.

Use `--episode-offset 5 --episodes 2` to render rows 5 and 6 from the original evaluation.
Use `--episodes 20` to film all twenty recorded episodes.
A one-episode sidecar score is compared with that episode's row, rather than the twenty-episode mean.

## 3. Render intermediate checkpoints

For another checkpoint, specify the `.pt` file directly and choose its evaluation clock:

```bash
python render_fj.py \
  --checkpoint "$RUN_DIR/policy_000500000.pt" \
  --clock irregular --split test --cohort iid --episodes 2 \
  --out videos/F_native_ct_sac/step500000 --gl egl
```

The renderer reads the saved configuration directly. It calls no configuration factory or training routine.
It supports `final.pt`, periodic `policy_*.pt`, and `latest.pt` when they contain the self-contained F/J checkpoint payload.
The actor and saved information map are sufficient for control; no `prepared.pt`, critic solve, replay buffer, or fitting pass is needed.
A `latest.pt` may be much larger because it also stores recovery data.
A checkpoint containing only a raw actor state dictionary needs the accompanying saved information map before it can be used here.

The same command supports the F/position observer: select that run's checkpoint.
Its causal GRU, normalization, reward/action history, actual duration, and queued next hold are restored from the saved contract.
The deterministic action matches `Agent.act(..., deterministic=True)` in the original evaluation.

## 4. Output and physical-time recording

Each selected episode produces an MP4 and a sidecar JSON.
The filename includes checkpoint label/hash, clock, split, cohort, and episode seed.
The sidecar records:

- The unchanged checkpoint configuration and identity.
- The selected episode index/seed and deterministic policy mode.
- Time-integrated score, raw decision return, decisions, physics steps, and horizon coverage.
- Original backend metadata and any current differences.
- Video frame times, physical simulation times, and capture delays.
- Differences from the selected original evaluation row, when available.
- Zero deployment/training updates and package versions.

Frames are scheduled at a uniform physical-time rate, 25 fps by default.
A recording hook reads the state at physics endpoints inside each action hold.
It does not advance the robot or sample an observation.
The irregular decision count therefore changes neither the playback speed nor the policy's information.
Capture time is quantized by the physics step; the sidecar reports that delay.
The exact backend, numerical stack, and hardware can still affect a long rollout.
Use `--strict-replay` to fail after saving the video when a complete reference episode differs beyond the recorded tolerances.

Use `--camera back`, `--fps 30`, `--width 800 --height 480`, or `--no-overlay` for presentation changes.
The renderer increases only the offscreen framebuffer size when needed.
Outputs are protected against overwriting; pass `--overwrite` for an intentional rerender.

## 5. Cheap previews, dry runs, and server launch

A three-second clip can still evaluate the complete episode and compare its final score:

```bash
python render_fj.py --checkpoint "$RUN_DIR/final.pt" \
  --evaluation-json "$RUN_DIR/evaluations/final_test_iid_fixed_n20.json" \
  --max-video-seconds 3 --out videos/short_clip
```

To stop simulation after the short clip, add `--stop-after-video`.
The report then labels the rollout partial and omits a complete-episode comparison.

```bash
python render_fj.py --checkpoint "$RUN_DIR/final.pt" \
  --clock fixed --max-video-seconds 1 --stop-after-video --out videos/precheck

# Inspect file hash and episode selection without Torch or MuJoCo imports.
python render_fj.py --checkpoint "$RUN_DIR/final.pt" \
  --evaluation-json "$RUN_DIR/evaluations/final_test_iid_fixed_n20.json" --dry-run
```

The shell launcher is equivalent:

```bash
CHECKPOINT="$RUN_DIR/final.pt" \
EVALUATION_JSON="$RUN_DIR/evaluations/final_test_iid_fixed_n20.json" \
EPISODES=1 VIDEO_DIR=videos/fixed MUJOCO_GL=egl \
  bash scripts/render_fj_v2.sh
```

On a compute node, run the same command under the site's allocated `srun` session.
This patch supplies no new training submission.
EGL needs access to the node's graphics driver. An OSMesa run needs the system OSMesa library.
An OpenGL failure stops with its error; it does not switch to a different renderer silently.

## 6. Validation and scope

See `VIDEO_PATCH_VALIDATION.md` for checks run during preparation.
The user's trained `.pt` file was not attached. Its actual trained-policy video must be generated on the machine holding that file.
The uploaded metric JSON alone contains no actor or observer weights.

Reference: [dm-control headless rendering backends](https://github.com/google-deepmind/dm_control#rendering).
