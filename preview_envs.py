"""Render contact environments without a display or any training update.

Rollout mode uses zero, random, sine, or a frozen checkpoint controller.
Tour mode moves only the camera and supplies a terrain-inspection video.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",default="prechecks/contact/environments.jsonl")
    p.add_argument("--out",default="prechecks/contact/videos")
    p.add_argument("--limit",type=int,default=3)
    p.add_argument("--mode",choices=["rollout","tour"],default="rollout")
    p.add_argument("--policy",choices=["zero","random","sine"],default="sine")
    p.add_argument("--checkpoint",help="Optional frozen F/J policy .pt; requires Torch")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--seconds",type=float,default=3.)
    p.add_argument("--fps",type=int,default=25)
    p.add_argument("--width",type=int,default=800)
    p.add_argument("--height",type=int,default=480)
    p.add_argument("--gl",choices=["egl","osmesa"],default="egl")
    p.add_argument("--clock",choices=["fixed","irregular"],default="fixed")
    p.add_argument("--start-x",type=float,default=4.,help="Root start for this audit; training samples its declared reset law")
    p.add_argument("--tour-end",type=float,default=20.)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.limit<1 or not 0<a.seconds<=30 or not 1<=a.fps<=60 or min(a.width,a.height)<128:
        p.error("Use a positive bounded preview budget (at most 30 s and 60 fps)")
    if a.width%2 or a.height%2:p.error("MP4 dimensions must be even")
    if a.checkpoint and a.mode!="rollout":p.error("A checkpoint is only used in rollout mode")
    if a.dry_run:
        print(json.dumps(dict(stage="preview",manifest=a.manifest,mode=a.mode,seconds=a.seconds,
            limit=a.limit,gl=a.gl,training_updates=0,expensive_work_launched=False),indent=2));return
    # Must be set before importing MuJoCo, dm_control, or PyOpenGL.
    os.environ["MUJOCO_GL"]=a.gl
    os.environ["PYOPENGL_PLATFORM"]=a.gl
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from fjbench.environment import Environment
    from fjbench.contact import query
    from fjbench.data import initial_history,next_history
    from bench.protocol import write_json,versions,digest
    from fjbench.config import make_config
    rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s.strip()][:a.limit]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError("Empty environment manifest")
    for row in rows:
        c=row["config"]
        if make_config(c["case"],c["experiment"],c["seed"],c)!=c or row.get("config_sha256")!=digest(c):
            raise ValueError("Stale or changed preview configuration; regenerate the manifest with sample_envs.py")
        env=Environment(c,row["reset_seed"],cohort=row["cohort"],clock=a.clock,
                        environment_spec=row["environment"],start_x=a.start_x)
        renderer=None;writer=None
        try:
            physics=env.base._env.physics
            physics.model.vis.global_.offwidth=max(a.width,int(physics.model.vis.global_.offwidth))
            physics.model.vis.global_.offheight=max(a.height,int(physics.model.vis.global_.offheight))
            renderer=mujoco.Renderer(physics.model.ptr,height=a.height,width=a.width)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth=90.;camera.elevation=-12.;camera.distance=6.
            options=mujoco.MjvOption()
            options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
            options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=True
            stem=f"{row['name']}_{a.mode}"
            writer=imageio.get_writer(out/(stem+".mp4"),fps=a.fps,codec="libx264",quality=7,
                                     macro_block_size=None)
            window=initial_history(env.observation,c["history"])
            agent=None
            if a.checkpoint:
                import torch
                from fjbench.calibration import load_prepared
                from fjbench.agent import Agent
                saved=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
                if saved["config"]!=c:raise ValueError("Checkpoint and manifest configuration differ")
                _,information,model,nodes,node_info=load_prepared(saved["public"],c,a.device)
                agent=Agent(c,saved["method"],information,model,nodes,node_info,a.device)
                agent.load(saved["agent"]);agent.actor.eval()
            rng=np.random.default_rng(row["reset_seed"]+131)
            phases=np.array([0.,.5,1.,np.pi,np.pi+.5,np.pi+1.])
            torso=physics.model.name2id("torso","body")
            diagnostics=[];frame_times=[];frame_index=0;ended=False
            totals=dict(audit_forward_solves=0,contact_pairs_checked=0,
                max_time_constant_error=0.,max_damping_ratio_error=0.,contact_point_work=0.,contact_slip_integral=0.,normal_impulse=0.,
                back_foot_normal_impulse=0.,front_foot_normal_impulse=0.,
                max_penetration=0.,max_effective_friction_error=0.)
            def frame(video_time):
                nonlocal frame_index
                if a.mode=="tour":
                    x=a.start_x+(a.tour_end-a.start_x)*video_time/a.seconds
                    camera.lookat[:]=[x,0,float(query(env.contact_map,[x])["height"][0])+.35]
                else:camera.lookat[:]=physics.data.xpos[torso]+np.array([.8,0.,0.])
                render_data=env.contact_audit.data if env.contact_audit.samples else physics.data.ptr
                renderer.update_scene(render_data,camera=camera,scene_option=options)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,a.width,65),fill=(15,25,38))
                status="CAMERA TOUR / NO PHYSICAL ROLLOUT" if a.mode=="tour" else f"{a.policy if agent is None else 'frozen checkpoint'} / NO TRAINING"
                draw.text((12,10),f"{row['name']} | {c['contact_profile']} | {status}",fill="white")
                draw.text((12,30),f"sim t={env.base.cur_t:.3f}s | frame t={video_time:.3f}s | contacts={len(env.contact_audit.records)} | penetration max={totals['max_penetration']:.4f}m",fill="#b8dcf4")
                draw.text((12,48),"Surface colour is an audit cue. Actor input uses the declared numerical sensor.",fill="#d0d0d0")
                image=np.asarray(im)
                writer.append_data(image)
                if frame_index==0:im.save(out/(stem+"_first.png"))
                frame_index+=1;frame_times.append(dict(frame=frame_index-1,video_time=video_time,simulation_time=env.base.cur_t))
            frame(0.)
            if a.mode=="tour":
                for index in range(1,int(a.seconds*a.fps)):frame(index/a.fps)
            else:
                while env.base.cur_t<a.seconds and not ended:
                    t=env.base.cur_t
                    if agent is not None:action=agent.act(window,env.g,env.hold(),t,True)
                    elif a.policy=="zero":action=np.zeros(6)
                    elif a.policy=="random":action=rng.uniform(-.4,.4,6)
                    else:action=.3*np.sin(2*np.pi*1.5*t+phases)
                    observation,info=env.step(action)
                    window=next_history(window,observation,action,info["r_src"],info["h"])
                    audit=info["contact_audit"]
                    diagnostics.append(dict(time=env.base.cur_t,**audit))
                    for key in totals:
                        totals[key]=max(totals[key],audit[key]) if key.startswith("max_") else totals[key]+audit[key]
                    while frame_index/a.fps<min(env.base.cur_t,a.seconds):frame(frame_index/a.fps)
                    ended=bool(info["terminated"] or info["truncated"])
            if totals["max_effective_friction_error"]>1e-7:
                raise RuntimeError("Effective contact friction differs from the ground patch")
            if max(totals["max_time_constant_error"],totals["max_damping_ratio_error"])>1e-7:
                raise RuntimeError("Effective contact response differs from the ground patch")
            report=dict(case=c["case"],mode=a.mode,policy=a.policy if agent is None else "frozen checkpoint",
                training_updates=0,backend=env.backend,versions=versions(),audit_truth=env.audit_truth(),
                elapsed_simulation=env.base.cur_t,requested_video_seconds=a.seconds,frames=frame_index,
                totals=totals,frame_times=frame_times,contacts=diagnostics,
                passed_pair_friction_check=(totals["max_effective_friction_error"]<=1e-7) if totals["contact_pairs_checked"] else None,
                passed_contact_response_check=True if totals["contact_pairs_checked"] else None,
                interpretation="Environment precheck; no trained-controller or success claim")
            write_json(out/(stem+"_audit.json"),report)
            print(json.dumps(dict(video=str((out/(stem+".mp4")).resolve()),frames=frame_index,
                training_updates=0,max_penetration=totals["max_penetration"],friction_error=totals["max_effective_friction_error"])))
        finally:
            if writer is not None:writer.close()
            if renderer is not None:renderer.close()
            env.close()


if __name__=="__main__":main()
