"""Checkpoint video regression tests. Run from the original project root."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from render_fj import FrameClock,select_episodes,validate_reference,compare_episode,file_sha256


class SelectionTests(unittest.TestCase):
    def test_recorded_seed_and_original_episode_index_are_retained(self):
        reference=dict(rows=[dict(episode_seed=17,g=10),dict(episode_seed=91,g=10)])
        choice=select_episodes(reference,1,1)[0]
        self.assertEqual((choice["index"],choice["seed"]),(1,91))
        with self.assertRaises(ValueError):select_episodes(reference,2,1)

    def test_reference_guards_checkpoint_method_and_calibration(self):
        saved=dict(config={"schema":2},method="ct_sac",calibration_sha256="cal")
        reference=dict(saved,checkpoint_sha256="abc",policy_mode="squashed_mean",deployment_updates=0)
        validate_reference(saved,reference,"abc")
        for key,value in (("checkpoint_sha256","other"),("method","ph_sac"),("calibration_sha256","other")):
            with self.assertRaises(ValueError):validate_reference(saved,dict(reference,**{key:value}),"abc")

    def test_frame_rate_follows_physical_time(self):
        clock=FrameClock(25,.2);targets=[]
        for t in (0.,.002,.004,.034,.064,.094,.124,.154,.184,.2):targets+=clock.due(t)
        self.assertEqual(len(targets),5)
        for a,b in zip(targets,(0.,.04,.08,.12,.16)):self.assertAlmostEqual(a,b)

    def test_partial_rollout_cannot_claim_full_reference_reproduction(self):
        observed=dict(time_score=1.,decisions=3)
        self.assertEqual(compare_episode(observed,observed,False)["status"],"partial_rollout")
        self.assertEqual(compare_episode(observed,observed,True)["status"],"matched")
        self.assertEqual(compare_episode(observed,dict(observed,time_score=2.),True)["status"],"mismatch")


TORCH=importlib.util.find_spec("torch") is not None
SIM=TORCH and importlib.util.find_spec("mujoco") is not None and importlib.util.find_spec("dm_control") is not None and importlib.util.find_spec("imageio_ffmpeg") is not None


@unittest.skipUnless(TORCH,"Torch required for actor checkpoint tests")
class ActorTests(unittest.TestCase):
    def test_actor_only_loader_matches_existing_Agent_act_for_F_sensors(self):
        import numpy as np
        import torch
        from fjbench.config import make_config
        from fjbench.models import InformationMap
        from fjbench.agent import Agent
        from fjbench.data import initial_history
        from render_fj import FrozenPolicy
        torch.set_num_threads(1)
        for sensor,experiment in (("native","native"),("position","position")):
            c=make_config("F",experiment,overrides={"hidden":[16,16]})
            # Exercise the original schema-2 information contract directly.
            c={k:v for k,v in c.items() if not k.startswith("contact_")};c["schema"]=2
            n=8 if sensor=="position" else 17
            info=InformationMap(c,np.zeros(n+8),np.ones(n+8))
            for method in ("ct_sac","ct_td3","ph_sac"):
                agent=Agent(c,method,info,None,None,None,"cpu")
                saved=dict(config=c,agent=agent.state(),public=dict(config=dict(c,phase_steps=c["phase_steps"]+1),
                    information=info.state_dict(),information_mean=np.zeros(n+8),information_scale=np.ones(n+8)))
                policy=FrozenPolicy(saved,"cpu");window=initial_history(np.ones(n,np.float32)*.1,c["history"])
                expected=agent.act(window,10.,.024,.1,True)
                actual=policy.act(window,10.,.024,.1)
                np.testing.assert_array_equal(actual,expected)
                self.assertFalse(policy.actor.training)
                self.assertTrue(all(not p.requires_grad for p in policy.actor.parameters()))
                bad=dict(saved,public=dict(saved["public"],config=dict(c,history=c["history"]+1)))
                with self.assertRaises(ValueError):FrozenPolicy(bad,"cpu")


@unittest.skipUnless(SIM,"MuJoCo/Torch/FFmpeg required for headless integration check")
class VideoTests(unittest.TestCase):
    def test_schema2_checkpoint_replay_keeps_scores_and_both_clocks(self):
        for experiment in ("native","position"):
            with self.subTest(experiment=experiment):self._check_schema2_replay(experiment)

    def _check_schema2_replay(self,experiment):
        os.environ["MUJOCO_GL"]="egl";os.environ["PYOPENGL_PLATFORM"]="egl"
        import numpy as np
        import torch
        import imageio.v2 as imageio
        from fjbench.config import make_config
        from fjbench.models import InformationMap
        from fjbench.agent import Agent
        from fjbench.environment import Environment
        from fjbench.data import initial_history,next_history
        from render_fj import FrozenPolicy
        torch.set_num_threads(1)
        c=make_config("F",experiment,overrides={"horizon":.2,"hidden":[16,16]})
        c={k:v for k,v in c.items() if not k.startswith("contact_")};c["schema"]=2
        width=(8 if experiment=="position" else 17)+8
        info=InformationMap(c,np.zeros(width),np.ones(width))
        agent=Agent(c,"ct_sac",info,None,None,None,"cpu")
        saved=dict(config=c,method="ct_sac",calibration_sha256="synthetic-test-calibration",agent=agent.state(),
            public=dict(config=c,information=info.state_dict(),information_mean=np.zeros(width),information_scale=np.ones(width)))
        policy=FrozenPolicy(saved)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/"fixture.pt";torch.save(saved,checkpoint)
            digest=file_sha256(checkpoint)
            for clock in ("fixed","irregular"):
                env=Environment(c,100000,0,"iid",clock)
                try:
                    window=initial_history(env.observation,c["history"]);raw=integral=elapsed=0.;decisions=physics=0
                    while True:
                        action=policy.act(window,env.g,env.hold(),env.base.cur_t)
                        observation,transition=env.step(action)
                        raw+=transition["r_src"];integral+=transition["reward_time_integral"];elapsed+=transition["h"]
                        decisions+=1;physics+=transition["physics_steps"]
                        window=next_history(window,observation,action,transition["r_src"],transition["h"])
                        if transition["terminated"] or transition["truncated"]:break
                    row=dict(episode_seed=100000,g=env.g,time_score=1000*integral/c["horizon"],
                        raw_decision_return=raw,elapsed_seconds=elapsed,decisions=decisions,physics_steps=physics)
                finally:env.close()
                reference=dict(config=c,method="ct_sac",checkpoint_sha256=digest,
                    calibration_sha256=saved["calibration_sha256"],clock=clock,cohort="iid",split="test",
                    policy_mode="squashed_mean",deployment_updates=0,rows=[row])
                reference_path=root/(clock+".json");reference_path.write_text(json.dumps(reference))
                command=[sys.executable,str(Path(__file__).resolve().parents[1]/"render_fj.py"),
                    "--checkpoint",str(checkpoint),"--evaluation-json",str(reference_path),
                    "--episodes","1","--out",str(root/clock),"--width","320","--height","240",
                    "--strict-replay"]
                process=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(process.returncode,0,process.stderr)
                report=json.loads(next((root/clock).glob("*.json")).read_text())
                self.assertEqual(report["reference_comparison"]["status"],"matched")
                self.assertEqual(report["config"],c);self.assertEqual(report["training_updates"],0)
                self.assertLessEqual(max(f["delay"] for f in report["frame_times"]),c["physics_dt"]+1e-8)
                with imageio.get_reader(report["video"]) as reader:
                    self.assertEqual(reader.count_frames(),5)
                    self.assertEqual(reader.get_data(0).shape,(240,320,3))
            self.assertEqual(file_sha256(checkpoint),digest)


if __name__=="__main__":unittest.main()
