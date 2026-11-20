"""Optional server preflight: two zero-action MuJoCo decisions per task."""
import argparse
import json
import numpy as np
from bench.env import make_common_env
from bench.protocol import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default=",".join(TASKS))
    a = p.parse_args()
    for task in a.tasks.split(","):
        env = make_common_env(task, 201, context="point_drag")
        try:
            obs, _ = env.reset(seed=201)
            for _ in range(2):
                obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
                assert np.isfinite(obs).all() and np.isfinite(reward)
                assert info["dt_used"] > 0 and info["directly_forced_dofs"] > 0
            print(json.dumps(dict(info, task=task, nq=env._env.physics.model.nq,
                                  nv=env._env.physics.model.nv, obs_dim=len(obs),
                                  native_dt=env.dt_default, force_body=env.force_body_name,
                                  force_point_local=env.force_point_local.tolist())))
        finally:
            env.close()


if __name__ == "__main__":
    main()
