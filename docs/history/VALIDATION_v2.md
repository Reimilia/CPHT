# v2 F/J preparation checks

Prepared on 22 October 2026.

No policy training, observer fitting, model fitting, or MuJoCo transition was executed locally.
The package contains scripts and experiment definitions, with no new control-performance result.

The test command found 34 tests: 30 passed and 4 were skipped because Torch is absent.
Python compilation, shell syntax checks, and dependency-free F/J train/prepare/eval previews passed.

The new checks cover:

- The four F/J experiments, fixed F parameters, and separate task/regime test values.
- Entropy units and the common-value actor-ranking identity.
- The exact OU environmental marginal step.
- Causal current/successor histories and episode/ring-buffer boundaries.
- Rejection of hidden physical parameters as public replay fields.
- Exclusion of calibration validation records through replay recovery and overwrite.
- Twin-min operator contraction with positive rows.
- Exact residual recovery for two distinct free critic heads.
- Physical-termination removal of future value and entropy.
- Midpoint numerical storage, inverse errors, finite residual-to-value bounds, and the cotangent derivative.
- The residual identity after a frozen operator changes.
- Preservation of the original source archive and published method/task configurations.

The optional Torch check includes positive conditioned neural rows and one synthetic update per adapter.
The remaining three optional Torch checks are retained from v1.
They have not been executed in this runtime.

The full supplied plans have 48 common preparation jobs and 384 training jobs.
They are instructions, rather than completed experiments.

Runtime checks remain for the compute server:

```bash
python -m unittest discover -s tests -v
python check_fj_backend.py
```

The backend check takes two zero-action decisions in each of the four F/J experiments.
It fits no observer, operator, or policy.

The simulator calls, payload/friction modifications, neural training, GPU execution, and complete resume path require runtime verification with the installed stack.
The retained XML/gear assertions and run manifests expose a backend mismatch.
The empirical observer checks provide no exact posterior or physical-parameter certificate.
The strict numerical mode concerns sampled finite critic coordinates.
The default empirical mode records fitting and returned-storage debt while retaining regression progress.
Neither mode establishes global neural convergence or policy optimality.

The supplied CT-RL source archive remains unchanged.
The legacy v1 implementation and its original validation record remain available.
