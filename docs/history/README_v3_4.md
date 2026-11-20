# MuJoCo benchmark v3.4: Pure CPHT feedback control

This release adds **pure_cpht**, an actor-free mechanical learning/control method. It fits a positive articulated Hamiltonian and a contact approximation, searches a periodic motion potential through the learned model, and computes six bounded motor commands through feedback. It trains no SAC actor, critic, or value function.

[PURE_CPHT.md](../PURE_CPHT.md) gives the equations, algorithm, input contract, server commands, deployment modes, and scope. The name denotes this new mechanical CPHT experiment. The manuscript's HJB/2PBVP value iteration remains a separate implementation.

Run commands from the extracted TPHT_CT_RL_Benchmark folder. The new launcher defaults to CPU and dry run:

~~~bash
CASE=both STAGE=prepare bash scripts/run_pure_cpht.sh
~~~

To prepare and train on your server:

~~~bash
export CASE=both
export METHODS=pure_cpht
export SEEDS=0
export TERRAIN=terrain
export MUJOCO_GL=egl

STAGE=prepare DRY_RUN=0 bash scripts/run_pure_cpht.sh
STAGE=train DRY_RUN=0 DEVICE=cpu bash scripts/run_pure_cpht.sh
~~~

The default budget is 100000 decisions, including 4096 calibration decisions. The learned mechanical model has 32 positive parameters; the contact approximation has four. Online physical-model refits occur every 512 control decisions. Motion planning uses a 19-parameter periodic potential. The guide includes a four-control-decision smoke preset.

Deploy the saved potential with feedback alone:

~~~bash
CASE=both METHODS=pure_cpht STAGE=eval DRY_RUN=0 \
  DEVICE=cpu FEEDBACK_ONLY=1 EPISODES=5 VIDEOS=1 \
  bash scripts/run_pure_cpht.sh
~~~

Set FEEDBACK_ONLY=0 to continue motion replanning during evaluation. Physical/contact model weights remain frozen in both modes. Evaluation saves JSON, speed/progress PNGs, headless MP4s, and prediction diagnostics. The shared eval_physical_fj.py accepts direct checkpoint paths and the --feedback-only flag.

The new preset adds the existing noisy height/slope probes to the previous exact state and force-port measurements. To compare CT-SAC under the same information and budget, set METHODS=ct_sac,pure_cpht with the same OPTIONS file. Earlier trained videos provide qualitative context; this matched comparison needs fresh runs.

F retains course seed 1. J samples fresh courses. The task remains a 20 m path at 3 m/s with a 12 s deadline. Use CASE_TEST=J and TERRAIN_TEST=steps after F training for frozen transfer.

A one-second feedback preview and a small model JSON are bundled in prechecks/pure_cpht/. Use preview_pure_cpht.py to run that check before training. It demonstrates short actuation under a seeded reference.

[VALIDATION.md](VALIDATION.md) records the executed checks and their limits. No costly training was run for this release. The short tests verify numerical/control behavior and supply no full-course locomotion result.

The existing SAC experiments remain available through scripts/run_physical_first.sh. Their original default settings, calibration IDs, run IDs, and checkpoint interfaces are preserved. [PHYSICAL_FIRST.md](../PHYSICAL_FIRST.md) and [README_v3_3.md](README_v3_3.md) describe those experiments. Earlier F/J value-solver commands remain in their versioned guides.
