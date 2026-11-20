"""Run released code with one added post-training CT save callback."""
import argparse
from pathlib import Path
import sys
from bench.protocol import source_root, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["ct", "discrete"], required=True)
    p.add_argument("--final", required=True)
    p.add_argument("--expected-budget", type=int, required=True)
    a, remaining = p.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    source_root()
    if a.family == "ct":
        import benchmarks.run_ct_rl as native
        from common.callbacks import CheckpointCallback

        class FinalSave(CheckpointCallback):
            def _on_training_end(self):
                super()._on_training_end()
                count = int(self.algorithm.num_timesteps)
                if count != a.expected_budget:
                    raise RuntimeError(f"Native learner completed {count} decisions; expected {a.expected_budget}")
                self.algorithm.save(a.final)
                write_json(Path(a.final).with_suffix(".json"), dict(decisions=count,
                    save_position="on_training_end, after final learning update",
                    algorithm_source_changed=False))

        # Only add a final I/O action. The source files, training iterations,
        # periodic checkpoints, and native evaluation calls remain in use.
        native.CheckpointCallback = FinalSave
    else:
        import benchmarks.run_discrete_rl as native
    sys.argv = [native.__file__, *remaining]
    native.main()


if __name__ == "__main__":
    main()
