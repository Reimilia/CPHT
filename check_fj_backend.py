"""Optional server check: two zero-action decisions per F/J experiment."""
import json
import numpy as np
from fjbench.config import make_config,EXPERIMENTS,observation_dim
from fjbench.environment import Environment


def main():
    for case,experiments in EXPERIMENTS.items():
        for experiment in experiments:
            c=make_config(case,experiment)
            env=Environment(c,201,0)
            try:
                expected=observation_dim(c)
                for _ in range(2):
                    observation,info=env.step(np.zeros(6,np.float32))
                    assert observation.shape==(expected,) and np.isfinite(observation).all()
                    assert info["h"]>0 and np.isfinite(info["R"])
                print(json.dumps(dict(case=case,experiment=experiment,observation_dim=expected,
                    backend=env.backend,public_info=info,audit_truth=env.audit_truth())))
            finally:env.close()


if __name__=="__main__":main()
