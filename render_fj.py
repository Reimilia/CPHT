"""Render an existing F/J checkpoint without training or configuration migration.

Copy this file into the original TPHT_CT_RL_Benchmark project root. It loads
only the saved actor and information map. No calibration file or critic solve
is needed. A supplied evaluation JSON pins checkpoint/configuration/episodes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def select_episodes(reference,episodes,offset,split="test",seed_start=None):
    if episodes<1 or offset<0:raise ValueError("Use a positive episode count and nonnegative offset")
    if reference is not None:
        rows=reference.get("rows",[])
        if offset+episodes>len(rows):raise ValueError("Requested episodes exceed the evaluation JSON rows")
        return [dict(index=i,seed=int(rows[i]["episode_seed"]),reference=rows[i])
                for i in range(offset,offset+episodes)]
    start=seed_start if seed_start is not None else (50000 if split=="validation" else 100000)
    return [dict(index=i,seed=start+i,reference=None) for i in range(offset,offset+episodes)]


def validate_reference(saved,reference,checkpoint_sha):
    if reference is None:return
    if reference.get("checkpoint_sha256")!=checkpoint_sha:
        raise ValueError("Checkpoint SHA-256 differs from the evaluation JSON. Select its exact .pt file, or render a different checkpoint without --evaluation-json.")
    if saved["config"]!=reference["config"] or saved["method"]!=reference["method"]:
        raise ValueError("Checkpoint configuration/method differs from the evaluation JSON")
    if reference.get("policy_mode")!="squashed_mean" or reference.get("deployment_updates",0)!=0:
        raise ValueError("This renderer reproduces frozen squashed-mean evaluation")
    if reference.get("calibration_sha256")!=saved.get("calibration_sha256"):
        raise ValueError("Checkpoint calibration identity differs from the evaluation JSON")


class FrozenPolicy:
    def __init__(self,saved,device="cpu"):
        import torch
        from fjbench.models import InformationMap
        from fjbench.config import calibration_key
        from bench.learner import Actor
        c=saved["config"];public=saved.get("public")
        if not public or "information" not in public:
            raise ValueError("The checkpoint needs its self-contained public information map; use final.pt, latest.pt, or a policy_*.pt from train_fj.py")
        if calibration_key(public.get("config",c))!=calibration_key(c):
            raise ValueError("Checkpoint public information contract differs from its saved configuration")
        self.device=torch.device(device)
        self.information=InformationMap(c,public["information_mean"],public["information_scale"]).to(self.device)
        self.information.load_state_dict(public["information"],strict=True)
        self.actor=Actor(self.information.feature_dim,6,c["hidden"]).to(self.device)
        self.actor.load_state_dict(saved["agent"]["actor"],strict=True)
        self.information.eval().requires_grad_(False);self.actor.eval().requires_grad_(False)

    def act(self,window,g,h,time):
        import torch
        with torch.inference_mode():
            inputs=[torch.as_tensor(v,dtype=torch.float32,device=self.device)
                    for v in (window[None],[[g]],[[h]],[[time]])]
            return self.actor.sample(self.information(*inputs),deterministic=True)[0][0].cpu().numpy()


def backend_differences(recorded,current):
    if not recorded:return {}
    keys=("xml_sha256","gears","nq","nv","nu","mujoco_integrator","force_body","force_point_local")
    return {key:dict(checkpoint=recorded[key],current=current.get(key))
            for key in keys if key in recorded and recorded[key]!=current.get(key)}


class FrameClock:
    """Sample uniform physical-time frames from physics-substep callbacks."""
    def __init__(self,fps,seconds):
        self.fps=fps;self.seconds=seconds;self.count=0

    def due(self,simulation_time):
        targets=[]
        while self.count/self.fps<self.seconds-1e-10 and self.count/self.fps<=simulation_time+1e-10:
            targets.append(self.count/self.fps);self.count+=1
        return targets


def compare_episode(observed,expected,full,rtol=1e-5,atol=1e-5):
    if expected is None:return None
    if not full:return dict(status="partial_rollout",score_comparison_performed=False)
    fields=("time_score","raw_decision_return","elapsed_seconds","decisions","physics_steps","g")
    difference={k:observed[k]-expected[k] for k in fields if k in expected}
    matched=all(math.isclose(observed[k],expected[k],rel_tol=rtol,abs_tol=atol) for k in difference)
    return dict(status="matched" if matched else "mismatch",score_comparison_performed=True,
                difference=difference,rtol=rtol,atol=atol)


def render_episode(saved,policy,choice,clock,cohort,split,args,checkpoint_sha):
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from fjbench.environment import Environment
    from fjbench.data import initial_history,next_history
    c=saved["config"]
    environment_cohort="validation" if split=="validation" else cohort
    env=Environment(c,choice["seed"],choice["index"],environment_cohort,clock)
    renderer=writer=None;original_force=None;success=False
    label=(args.checkpoint_label or Path(args.checkpoint).stem)
    label="".join(x if x.isalnum() or x in "_-" else "_" for x in label)
    name=f"{label}_{checkpoint_sha[:10]}_{clock}_{split}_{cohort}_seed{choice['seed']}"
    video=Path(args.out)/(name+".mp4");report_path=Path(args.out)/(name+".json")
    temporary=video.with_name(video.stem+".part.mp4")
    try:
        if not args.overwrite and any(p.exists() for p in (video,report_path,temporary)):
            raise FileExistsError(f"Output exists: {video}. Choose another --out or pass --overwrite.")
        Path(args.out).mkdir(parents=True,exist_ok=True)
        mismatch=backend_differences(saved["public"].get("backend"),env.backend)
        if mismatch and not args.allow_backend_mismatch:
            raise ValueError("Simulator model differs from checkpoint backend: "+json.dumps(mismatch)+
                             ". Use the training simulator versions, or pass --allow-backend-mismatch for a labelled diagnostic run.")
        if choice["reference"] and not math.isclose(env.g,choice["reference"]["g"],abs_tol=1e-6):
            raise ValueError("Reset task descriptor differs from recorded episode")
        p=env.base._env.physics
        p.model.vis.global_.offwidth=max(args.width,int(p.model.vis.global_.offwidth))
        p.model.vis.global_.offheight=max(args.height,int(p.model.vis.global_.offheight))
        renderer=mujoco.Renderer(p.model.ptr,height=args.height,width=args.width)
        writer=imageio.get_writer(temporary,fps=args.fps,codec="libx264",quality=7,macro_block_size=None)
        maximum=min(c["horizon"],args.max_video_seconds or c["horizon"])
        frame_clock=FrameClock(args.fps,maximum);frame_times=[]
        camera=int(args.camera) if args.camera.isdecimal() else args.camera
        def capture(t):
            for target in frame_clock.due(t):
                renderer.update_scene(p.data.ptr,camera=camera)
                pixels=renderer.render()
                if not args.no_overlay:
                    im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,args.width,44),fill=(15,25,38))
                    draw.text((10,7),f"{c['case']}/{c['experiment']} | {saved['method']} | {clock} | seed {choice['seed']}",fill="white")
                    draw.text((10,25),f"physical t={t:.3f}s | frozen checkpoint | no learning updates",fill="#c4deef")
                    pixels=np.asarray(im)
                writer.append_data(pixels)
                frame_times.append(dict(frame=len(frame_times),frame_time=target,simulation_time=float(t),delay=float(t-target)))
        capture(0.)
        # The v2 wrapper calls _point_force once at each left physics endpoint.
        # Rendering reads state and consumes no robot, wind, sensor, or policy RNG.
        original_force=env.base._point_force
        def recording_force(*a,**kw):
            capture(env.base.cur_t+env.base._micro_elapsed)
            return original_force(*a,**kw)
        env.base._point_force=recording_force
        window=initial_history(env.observation,c["history"])
        raw=integral=elapsed=work=0.;decisions=physical=forced=0;full=False
        while True:
            action=policy.act(window,env.g,env.hold(),env.base.cur_t)
            if action.shape!=(6,) or not np.isfinite(action).all():raise FloatingPointError("Invalid checkpoint action")
            observation,info=env.step(action)
            raw+=info["r_src"];integral+=info["reward_time_integral"];elapsed+=info["h"]
            work+=info["external_work"];decisions+=1;physical+=info["physics_steps"]
            forced=max(forced,info["directly_forced_dofs"])
            window=next_history(window,observation,action,info["r_src"],info["h"])
            capture(env.base.cur_t)
            if info["terminated"] or info["truncated"]:full=True;break
            if args.stop_after_video and env.base.cur_t>=maximum-1e-10:break
        observed=dict(episode_seed=choice["seed"],g=env.g,time_score=1000*integral/c["horizon"],
            raw_decision_return=raw,elapsed_seconds=elapsed,horizon_coverage=elapsed/c["horizon"],
            decisions=decisions,physics_steps=physical,external_work=work,directly_forced_dofs=forced,
            audit_truth=env.audit_truth())
        comparison=compare_episode(observed,choice["reference"],full)
        # Close and atomically install the completed MP4 before recording success.
        writer.close();writer=None
        temporary.replace(video)
        versions={}
        import importlib.metadata
        for package in ("torch","mujoco","dm-control","imageio","imageio-ffmpeg"):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]=None
        report=dict(config=c,method=saved["method"],checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_sha256=checkpoint_sha,calibration_sha256=saved.get("calibration_sha256"),
            clock=clock,cohort=cohort,split=split,episode_index=choice["index"],policy_mode="squashed_mean",
            deployment_updates=0,training_updates=0,actor_and_observer_only=True,full_episode=full,
            observed=observed,reference_comparison=comparison,backend=env.backend,backend_differences=mismatch,
            versions=versions,torch_device=str(policy.device),render_backend=args.gl,video=str(video.resolve()),
            frames=len(frame_times),fps=args.fps,video_seconds=len(frame_times)/args.fps,
            capture_mode="uniform_physical_time_on_physics_endpoints",frame_times=frame_times)
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        success=True
        return report
    finally:
        if original_force is not None:env.base._point_force=original_force
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        env.close()
        if not success and temporary.exists():temporary.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,help="Existing final.pt, latest.pt, or policy_*.pt")
    p.add_argument("--evaluation-json","--from-eval",dest="evaluation_json",help="Pins the checkpoint SHA, configuration, clock, cohort, and episode rows")
    p.add_argument("--clock",choices=["fixed","irregular"],help="Without a JSON, defaults to irregular")
    p.add_argument("--cohort",help="Without a JSON, defaults to iid")
    p.add_argument("--split",choices=["test","validation"],help="Without a JSON, defaults to test")
    p.add_argument("--seed-start",type=int,help="Without a JSON, defaults to 100000 (test) or 50000 (validation)")
    p.add_argument("--episodes",type=int,default=1)
    p.add_argument("--episode-offset",type=int,default=0)
    p.add_argument("--out",default="videos/fj")
    p.add_argument("--checkpoint-label")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--torch-threads",type=int,default=1)
    p.add_argument("--gl",choices=["egl","osmesa"],default="egl")
    p.add_argument("--fps",type=int,default=25)
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    p.add_argument("--camera",default="side",help="Cheetah camera name side/back, or a numeric camera ID")
    p.add_argument("--max-video-seconds",type=float,help="Limit the saved clip; evaluation still completes unless --stop-after-video is set")
    p.add_argument("--stop-after-video",action="store_true",help="Cheap partial-rollout preview; disables full-episode score comparison")
    p.add_argument("--allow-backend-mismatch",action="store_true")
    p.add_argument("--strict-replay",action="store_true",help="Fail after rendering if a full reference replay differs numerically")
    p.add_argument("--no-overlay",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--dry-run",action="store_true",help="Inspect paths and episode selection without loading Torch or MuJoCo")
    a=p.parse_args()
    if not Path(a.checkpoint).is_file():p.error(f"Checkpoint does not exist: {a.checkpoint}")
    if not 1<=a.fps<=120 or min(a.width,a.height)<128 or a.width%2 or a.height%2:p.error("Use 1..120 fps and even image dimensions >=128")
    if a.torch_threads<1 or (a.max_video_seconds is not None and a.max_video_seconds<=0):p.error("Positive threads and clip duration required")
    if a.stop_after_video and a.max_video_seconds is None:p.error("--stop-after-video requires --max-video-seconds")
    if a.stop_after_video and a.strict_replay:p.error("Strict replay requires complete evaluation episodes")
    reference=json.loads(Path(a.evaluation_json).read_text()) if a.evaluation_json else None
    if reference:
        if a.seed_start is not None:p.error("Episode seeds come from the evaluation JSON; omit --seed-start")
        for key in ("clock","cohort","split"):
            value=getattr(a,key)
            if value is not None and value!=reference[key]:p.error(f"--{key} conflicts with the evaluation JSON")
    clock=a.clock or (reference["clock"] if reference else "irregular")
    cohort=a.cohort or (reference["cohort"] if reference else "iid")
    split=a.split or (reference["split"] if reference else "test")
    try:choices=select_episodes(reference,a.episodes,a.episode_offset,split,a.seed_start)
    except ValueError as exc:p.error(str(exc))
    checkpoint_sha=file_sha256(a.checkpoint)
    if reference and checkpoint_sha!=reference.get("checkpoint_sha256"):
        p.error("Checkpoint SHA-256 differs from the evaluation JSON. Select its exact checkpoint, or omit --evaluation-json for another checkpoint.")
    if a.dry_run:
        print(json.dumps(dict(checkpoint=a.checkpoint,checkpoint_sha256=checkpoint_sha,clock=clock,cohort=cohort,
            split=split,episode_indices=[r["index"] for r in choices],episode_seeds=[r["seed"] for r in choices],
            output=a.out,render_backend=a.gl,training_updates=0,expensive_work_launched=False),indent=2));return
    # Set the headless backend before importing simulator-dependent modules.
    os.environ["MUJOCO_GL"]=a.gl;os.environ["PYOPENGL_PLATFORM"]=a.gl
    import torch
    torch.set_num_threads(a.torch_threads)
    if a.device=="cuda" and not torch.cuda.is_available():p.error("CUDA policy inference requested but CUDA is unavailable; use --device cpu")
    saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    validate_reference(saved,reference,checkpoint_sha)
    c=saved["config"]
    if c.get("schema") not in (2,3,4):p.error("Expected a schema-2/3/4 train_fj.py checkpoint")
    if c["case"]=="F" and cohort!="iid":p.error("F evaluates its fixed plant with cohort iid")
    policy=FrozenPolicy(saved,a.device)
    for choice in choices:
        result=render_episode(saved,policy,choice,clock,cohort,split,a,checkpoint_sha)
        status=(result["reference_comparison"] or {}).get("status","no_reference")
        print(json.dumps(dict(video=result["video"],episode_seed=choice["seed"],clock=clock,
            time_score=result["observed"]["time_score"],frames=result["frames"],reference_status=status,training_updates=0)))
        if a.strict_replay and status!="matched":raise RuntimeError("Video saved, but the full reference replay did not match. Inspect the sidecar JSON and simulator versions.")


if __name__=="__main__":main()
