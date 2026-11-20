"""Finite forward course, speed tracking, and physical-time task diagnostics."""
import math
import numpy as np


class PathTask:
    def __init__(self,c,target,initial_x):
        self.c=c;self.target=float(target);self.initial_x=float(initial_x)
        self.goal=c["path_start_x"]+c["path_length"]
        if initial_x>=self.goal:raise ValueError("Path reset is already beyond the goal")
        self.elapsed=self.squared_error=self.absolute_error=self.in_band=self.effort=self.cost=0.
        self.reason=None;self.event_paid=False;self.event_return=0.;self.last_x=initial_x;self.last_v=0.
        self.trace=[];self.next_trace=0.
        self.last_clearance=0.

    def seed_trace(self,speed,clearance):
        self.last_v=float(speed);self.last_clearance=float(clearance)
        self._append_trace();self.next_trace=self.c["path_trace_dt"]

    def _append_trace(self):
        self.trace.append(dict(time=self.elapsed,x=self.last_x,remaining=max(0.,self.goal-self.last_x),
            speed=self.last_v,target_speed=self.target,clearance=self.last_clearance))

    def cost_rate(self,speed,action):
        relative=(speed-self.target)/self.target
        return (self.c["path_speed_weight"]*relative**2+
            self.c["path_effort_weight"]*float(np.mean(np.asarray(action)**2))+
            self.c["path_time_weight"]+self.c["path_backwards_weight"]*max(-speed/self.target,0.))

    def condition(self,x,clearance,pitch):
        if clearance<self.c["path_fall_clearance"] or abs(pitch)>self.c["path_fall_pitch"]:return "fall"
        if x>=self.goal:return "goal"
        return None

    def record(self,x,speed,action,dt,clearance,pitch):
        error=speed-self.target
        self.elapsed+=dt;self.squared_error+=error*error*dt;self.absolute_error+=abs(error)*dt
        self.in_band+=float(abs(error)<=self.c["path_speed_band"]*self.target)*dt
        self.effort+=float(np.mean(np.asarray(action)**2))*dt
        self.cost+=self.cost_rate(speed,action)*dt
        self.last_x=float(x);self.last_v=float(speed)
        self.last_clearance=float(clearance)
        self.reason=self.condition(x,clearance,pitch)
        if not self.trace or self.elapsed>=self.next_trace-1e-9 or self.reason:
            self._append_trace()
            self.next_trace=self.elapsed+self.c["path_trace_dt"]

    def close_interval(self,truncated=False):
        if truncated and self.reason is None:self.reason="timeout"
        if self.reason and not self.event_paid:
            if not self.trace or abs(self.trace[-1]["time"]-self.elapsed)>1e-10:self._append_trace()
            self.event_paid=True
            self.event_return=self.c["path_goal_bonus"] if self.reason=="goal" else -self.c["path_failure_penalty"]
            if self.reason=="fall":
                # Charge the remaining stationary failure cost as an absorbing
                # tail, so an early fall cannot avoid the course running cost.
                beta=-math.log(self.c["gamma_reference"])/self.c["reference_time"]
                remaining=max(0.,self.c["horizon"]-self.elapsed)
                stall=self.c["path_speed_weight"]+self.c["path_time_weight"]
                self.event_return-=stall*(-math.expm1(-beta*remaining))/beta
            return self.event_return
        return 0.

    def summary(self,include_trace=True):
        t=max(self.elapsed,1e-12);success=self.reason=="goal"
        distance=max(self.goal-self.initial_x,1e-12)
        result=dict(success=success,terminal_reason=self.reason or "running",target_speed=self.target,
            initial_x=self.initial_x,goal_x=self.goal,path_length=self.c["path_length"],
            initial_remaining_distance=distance,completed_fraction=float(np.clip((self.last_x-self.initial_x)/distance,0,1)),
            distance_travelled=self.last_x-self.initial_x,remaining_distance=max(0.,self.goal-self.last_x),
            elapsed_seconds=self.elapsed,completion_time=self.elapsed if success else None,
            ideal_completion_time=distance/self.target,
            arrival_time_error=self.elapsed-distance/self.target if success else None,
            speed_rmse=math.sqrt(self.squared_error/t),speed_mae=self.absolute_error/t,
            relative_speed_rmse=math.sqrt(self.squared_error/t)/self.target,
            speed_band_fraction=self.in_band/t,mean_squared_command=self.effort/t,
            integrated_running_cost=self.cost,undiscounted_task_return=-self.cost+self.event_return,
            fall=self.reason=="fall",timeout=self.reason=="timeout")
        if include_trace:result["trajectory"]=self.trace.copy()
        return result
