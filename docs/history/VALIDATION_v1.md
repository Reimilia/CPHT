# Preparation checks

Prepared on 20 October 2026.

No neural training, complete MuJoCo episode, or benchmark sweep was run locally.
The package provides a protocol and executable experiment scripts.
It contains no measured control-performance result.

The following checks passed:

- Four robotics task identifiers and all 32 published top method/task CSV rows resolve from the supplied source archive.
- The full plans contain 384 released runs and 240 matched runs.
- Train and evaluation dry runs resolve the intended budgets, methods, and paths without importing Torch or MuJoCo.
- Python compilation and shell syntax checks pass.
- The finite operator passes probability, contraction, residual recovery, cotangent, midpoint storage, and residual-to-value-bound checks.
- CLI integration checks confirm aggregation over independent training seeds, visible missing results, validation-only selection, and disjoint tuning/test training seeds.

`python -m unittest discover -s tests -v` found 17 tests.
Fourteen tests passed. Three optional Torch tests were skipped because Torch is absent from the preparation runtime.
Those tests check agreement between Torch and NumPy solver arithmetic, one update of each neural adapter, and replay-state recovery.

The simulator, neural adapters, CUDA execution, callback integration with the full installed baseline stack, and full resume path require server runtime checks.
Use the supplied short checks before submitting a large array:

```bash
python -m unittest discover -s tests -v
python check_backend.py
```

The first command performs small algebra and synthetic-batch checks.
The second command takes two physics-backed decisions per task.
Neither command trains a control policy.

Source archive SHA-256:

```text
152162e3633cd634530efb9cc007c7d014cb145a4bbcbdc93b4d141f6d299cbc
```

The original source remains unchanged.
The wrapper adds a CT save callback after the final learning update and preserves the native periodic checkpoint.
The matched collocation algorithm is an experimental extension and has a separate method label.
