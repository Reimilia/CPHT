"""Read-only physical-time recording shared by checkpoint evaluation."""
from pathlib import Path
import json


class FrameClock:
    """Sample uniform physical-time frames from physics-substep callbacks."""
    def __init__(self,fps,seconds):
        self.fps=fps;self.seconds=seconds;self.count=0

    def due(self,simulation_time):
        targets=[]
        while self.count/self.fps<self.seconds-1e-10 and self.count/self.fps<=simulation_time+1e-10:
            targets.append(self.count/self.fps);self.count+=1
        return targets


class EpisodeRecorder:
    def __init__(self,env,path,options,label=""):
        import mujoco
        import imageio.v2 as imageio
        self.env=env;self.path=Path(path);self.options=options;self.label=label
        camera=options.get("camera","side")
        self.camera=int(camera) if isinstance(camera,str) and camera.isdecimal() else camera
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists() and not options.get("overwrite",False):raise FileExistsError(str(self.path))
        self.temporary=self.path.with_name(self.path.stem+".part.mp4")
        p=env.base._env.physics
        width,height=options.get("width",640),options.get("height",480)
        p.model.vis.global_.offwidth=max(width,int(p.model.vis.global_.offwidth))
        p.model.vis.global_.offheight=max(height,int(p.model.vis.global_.offheight))
        self.renderer=mujoco.Renderer(p.model.ptr,width=width,height=height)
        self.writer=imageio.get_writer(self.temporary,fps=options.get("fps",25),codec="libx264",quality=7,macro_block_size=None)
        self.clock=FrameClock(options.get("fps",25),min(options.get("seconds") or env.config["horizon"],env.config["horizon"]))
        self.times=[];self.original=env.base._point_force;self.closed=False
        def recording_force(*a,**kw):
            self.capture(env.base.cur_t+env.base._micro_elapsed)
            return self.original(*a,**kw)
        env.base._point_force=recording_force
        self.capture(0.)

    def capture(self,time):
        from PIL import Image,ImageDraw
        import numpy as np
        for target in self.clock.due(time):
            self.renderer.update_scene(self.env.base._env.physics.data.ptr,camera=self.camera)
            im=Image.fromarray(self.renderer.render());draw=ImageDraw.Draw(im)
            width=self.options.get("width",640)
            draw.rectangle((0,0,width,48),fill=(15,25,38))
            draw.text((10,7),f"{self.label} | t={time:.3f}s | frozen policy",fill="white")
            if self.env.path_task is not None:
                x,v,_,_=self.env._path_state();task=self.env.path_task
                draw.text((10,27),f"speed {v:.2f} / target {task.target:.2f} m/s | remaining {max(0,task.goal-x):.2f} m",fill="#c4deef")
            self.writer.append_data(np.asarray(im))
            self.times.append(dict(frame=len(self.times),frame_time=target,simulation_time=float(time),delay=float(time-target)))

    def close(self,success=True,episode=None):
        if self.closed:return
        self.closed=True;self.env.base._point_force=self.original
        held=0
        if success and episode is not None:
            held=round(self.options.get("end_card_seconds",1.)*self.clock.fps)
            if held:
                from PIL import Image,ImageDraw
                import numpy as np
                self.renderer.update_scene(self.env.base._env.physics.data.ptr,camera=self.camera)
                im=Image.fromarray(self.renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,self.options.get("width",640),70),fill=(15,25,38))
                task=episode.get("path_metrics")
                result=task["terminal_reason"].upper() if task else "EPISODE ENDED"
                draw.text((10,9),f"{self.label} | {result}",fill="white")
                if task:draw.text((10,29),f"completion {task['completed_fraction']:.1%} | speed RMSE {task['speed_rmse']:.3f} m/s",fill="#c4deef")
                draw.text((10,49),"Terminal state held for display; evaluation has ended.",fill="#c4deef")
                for _ in range(held):self.writer.append_data(np.asarray(im))
        try:self.writer.close()
        finally:self.renderer.close()
        if success:
            self.temporary.replace(self.path)
            report=dict(video=str(self.path.resolve()),frames=len(self.times)+held,physical_frames=len(self.times),
                terminal_hold_frames=held,fps=self.clock.fps,video_seconds=(len(self.times)+held)/self.clock.fps,
                frame_times=self.times,episode=episode,training_updates=0)
            self.path.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            return report
        self.temporary.unlink(missing_ok=True)
        return None
