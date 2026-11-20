"""Latent periodic potential search and bounded feedback. No actor or critic."""
import math
import numpy as np
import torch
from .mechanics import Robot
from .contact import Contact,terrain
from .feedback import bounded_quadratic


class Controller:
    def __init__(self,model,options,neutral,seed=0,device='cpu',gait=None,feedback_only=False):
        self.robot=Robot(model['geometry'],model['parameters'],device=device)
        self.contact=Contact(self.robot,model['contact'],options);self.options=options
        self.generator=torch.Generator(device=device);self.generator.manual_seed(int(seed))
        self.neutral=self.robot.t(neutral);self.phase=0.;self.last_time=None;self.last_plan=-float('inf')
        self.feedback_only=feedback_only
        self.previous=self.robot.t(np.zeros(6));self.plans=0;self.predicted_steps=0
        self.force_correction=torch.zeros((len(self.robot.point_body),2),device=device)
        o=options
        self.current_port=self.robot.t(np.zeros(9));self.first_hold=o['planning_dt']
        seed_scale=o.get('gait_seed_scale',1.)
        self.mean=self.robot.t(gait) if gait is not None else torch.cat((self.neutral,
            seed_scale*self.robot.t([.25,-.35,.2,-.25,.35,-.2]),seed_scale*self.robot.t([0.,.12,-.15,0.,-.12,.15]),self.robot.t([o['gait_frequency']])))
        self.scale=self.robot.t([.08]*6+[o['gait_std']]*12+[.3])
        self.low=torch.cat((self.robot.limits[:,0]+.03,self.robot.t([-1.]*12+[o['frequency_min']])))
        self.high=torch.cat((self.robot.limits[:,1]-.03,self.robot.t([1.]*12+[o['frequency_max']])))
        self.mean=self.mean.clamp(self.low,self.high);self.diagnostics={}

    def references(self,parameters,phase):
        omega=2*math.pi*parameters[:,18]
        angle=phase if torch.is_tensor(phase) else self.robot.t(phase)
        if angle.ndim==0:angle=angle.expand(len(parameters))
        sine,cosine=torch.sin(angle)[:,None],torch.cos(angle)[:,None]
        wave=parameters[:,6:12]*sine+parameters[:,12:18]*cosine
        q=parameters[:,:6]+wave
        v=omega[:,None]*(parameters[:,6:12]*cosine-parameters[:,12:18]*sine)
        a=-omega[:,None]**2*wave
        limited=(q>self.robot.limits[:,1]-.02)|(q<self.robot.limits[:,0]+.02)
        q=q.clamp(self.robot.limits[:,0]+.02,self.robot.limits[:,1]-.02)
        return q,torch.where(limited,torch.zeros_like(v),v),torch.where(limited,torch.zeros_like(a),a)

    def feedback(self,q,v,port,scan,parameters,phase,target,quantities=None):
        r=self.robot;o=self.options;d=quantities if quantities is not None else r.quantities(q,v)
        qr,vr,ar=self.references(parameters,phase)
        root_z=q[:,1]+r.positions[1,1];height,_=terrain(scan,q[:,:1]+r.positions[1,0])
        desired=torch.cat((
            (o['speed_kd']*(target-d['com_velocity'][:,0]))[:,None],
            (o['height_kp']*(o['target_clearance']-(root_z-height[:,0]))-o['height_kd']*v[:,1])[:,None],
            (-o['pitch_kp']*q[:,2]-o['pitch_kd']*v[:,2])[:,None],
            ar+o['joint_kp']*(qr-q[:,3:])+o['joint_kd']*(vr-v[:,3:])),1)
        A=torch.linalg.solve(d['M'],r.B[None].expand(len(q),-1,-1))
        free=torch.linalg.solve(d['M'],(port-d['c']).unsqueeze(-1)).squeeze(-1)
        weights=r.t([1.,4.,4.,.15,.15,.15,.15,.15,.15])
        Q=A.transpose(1,2)@(weights[None,:,None]*A)+o['feedback_ridge']*torch.eye(6,device=q.device)[None]
        rhs=(A.transpose(1,2)@(weights*(desired-free))[:,:,None]).squeeze(-1)
        action,residual,_=bounded_quadratic(Q,rhs,o['feedback_iterations'])
        return action,residual

    def rollout(self,q0,v0,scan,parameters,target,phase):
        o=self.options;r=self.robot;batch=len(parameters)
        q=q0.expand(batch,-1).clone();v=v0.expand(batch,-1).clone()
        scan=scan[None].expand(batch,-1,-1);previous=self.previous[None].expand(batch,-1).clone()
        start=r.quantities(q,v)['com_position'][:,0];cost=torch.zeros(batch,device=q.device)
        invalid=torch.zeros(batch,device=q.device,dtype=torch.bool);phase=torch.full((batch,),phase,device=q.device)
        micro=o['planning_dt']/o['substeps'];elapsed=0.;last_velocity=torch.zeros(batch,device=q.device)
        for interval in range(o['horizon_steps']):
            # The first held input uses the measured requested duration.
            # Future input intervals use the declared nominal planning clock.
            duration=self.first_hold if interval==0 else o['planning_dt']
            steps=max(1,int(math.ceil(duration/micro-1e-9)));dt=duration/steps
            d=r.quantities(q,v)
            if interval==0:port=self.current_port[None].expand(batch,-1)
            else:port=self.contact.solve(q,v,previous,scan,d,self.force_correction,elapsed)['port']
            u,_=self.feedback(q,v,port,scan,parameters,phase,target,d)
            # Hold this command across the internal integration substeps.
            for substep in range(steps):
                if substep:d=r.quantities(q,v)
                applied=self.contact.solve(q,v,u,scan,d,self.force_correction,elapsed)
                acceleration=torch.linalg.solve(d['M'],(u@r.B.T+applied['port']-d['c'])[:,:,None]).squeeze(-1)
                v=v+dt*acceleration;q=q+dt*v
                root_height,_=terrain(scan,q[:,:1]+r.positions[1,0]);clearance=q[:,1]+r.positions[1,1]-root_height[:,0]
                speed=d['com_velocity'][:,0];last_velocity=speed
                stage=o['speed_weight']*(speed-target)**2+o['clearance_weight']*(clearance-o['target_clearance'])**2
                stage=stage+o['pitch_weight']*q[:,2]**2+o['effort_weight']*u.square().sum(1)
                outside=((d['points'][:,:,0]<scan[:,:1,0])|(d['points'][:,:,0]>scan[:,-1:,0])).float().mean(1)
                stage=stage+o['uncertainty_weight']*outside;cost=cost+dt*stage
                invalid|=(~torch.isfinite(q).all(1))|(~torch.isfinite(v).all(1))|(v.abs().max(1).values>80)|(clearance<.15)|(q[:,2].abs()>1.6)
                invalid|=((q[:,3:]<r.limits[:,0]-.3)|(q[:,3:]>r.limits[:,1]+.3)).any(1)
                q=torch.nan_to_num(q).clamp(-100,100);v=torch.nan_to_num(v).clamp(-100,100)
                phase=phase+2*math.pi*parameters[:,18]*dt;elapsed+=dt;self.predicted_steps+=batch
            previous=u
        end=r.quantities(q,v)['com_position'][:,0]
        # Progress beyond the requested distance receives no overspeed bonus.
        shortfall=(target*elapsed-(end-start)).clamp_min(0)
        cost=cost+o['progress_weight']*shortfall+.1*(last_velocity-target)**2
        return torch.nan_to_num(cost,nan=1e9,posinf=1e9,neginf=1e9)+invalid.float()*1e6,invalid

    def plan(self,q,v,scan,target):
        o=self.options;mean=self.mean.clone();std=self.scale.clone();best=mean.clone();best_cost=float('inf');fraction=0.
        for _ in range(o['iterations']):
            population=mean[None]+torch.randn((o['candidates'],19),generator=self.generator,device=q.device)*std
            population=population.clamp(self.low,self.high);population[0]=best
            cost,invalid=self.rollout(q,v,scan,population,target,self.phase)
            elite=torch.topk(-cost,o['elites']).indices
            if float(cost[elite[0]])<best_cost:best_cost=float(cost[elite[0]]);best=population[elite[0]].clone()
            mean=.25*mean+.75*population[elite].mean(0)
            std=(.25*std+.75*population[elite].std(0,unbiased=False)).clamp_min(.015)
            fraction=float(invalid.float().mean())
        self.mean=best;self.plans+=1
        self.diagnostics.update(planner_cost=best_cost,planner_invalid_fraction=fraction,control_optimizations=self.plans,
            predictive_substeps=self.predicted_steps,frequency_hz=float(self.mean[-1]))

    @torch.no_grad()
    def act(self,packet,target,hold,time):
        if 'scan' not in packet:raise ValueError('Pure CPHT needs measured terrain or a causal original-state terrain estimate')
        if 'environment' in packet:
            # This is an estimated environment state, separate from learned
            # mechanical/contact calibration weights stored in the checkpoint.
            self.contact.parameters[3]=float(packet['environment']['friction'])
            self.diagnostics.update(environment_belief=packet['environment'],
                information_mode=packet['information_contract']['mode'])
        if self.last_time is not None:self.phase=(self.phase+2*math.pi*float(self.mean[-1])*max(0.,time-self.last_time))%(2*math.pi)
        self.last_time=time;r=self.robot
        q=r.t(packet['raw'][:9])[None];v=r.t(packet['raw'][9:18])[None];port=r.t(packet['raw'][18:27])[None];scan=r.t(packet['scan'])
        self.current_port=port[0].clone();self.first_hold=float(hold)
        if not math.isfinite(self.first_hold) or self.first_hold<=0:raise ValueError('Positive finite requested control duration required')
        correction,observer=self.contact.observe(q,v,self.previous[None],scan,port)
        self.force_correction=correction[0]
        self.diagnostics.update({k:float(value[0]) for k,value in observer.items()})
        if not self.feedback_only and time-self.last_plan>=self.options['replan_seconds']-1e-9:
            self.plan(q,v,scan,target);self.last_plan=time
        action,residual=self.feedback(q,v,port,scan,self.mean[None],self.phase,target)
        self.previous=action[0].clone();d=r.quantities(q,v)
        self.diagnostics.update(feedback_projected_residual=float(residual[0]),action_saturation=float((abs(action)>.99).float().mean()),
            mechanical_energy=float(d['H'][0]),actuator_power=float((v[0,3:]*r.gears*action[0]).sum()),
            environment_power=float((v*port).sum()),damping_power=float(d['dissipation'][0]),phase=self.phase,
            actor_updates=0,critic_updates=0,control_optimizations=self.plans,feedback_only=self.feedback_only)
        return action[0].cpu().numpy()

    @torch.no_grad()
    def prediction_error(self,before,after,action,duration,physics_dt):
        r=self.robot;q=r.t(before['raw'][:9])[None];v=r.t(before['raw'][9:18])[None]
        scan=r.t(before['scan']);u=r.t(action)[None]
        initial=float(r.quantities(q,v)['H'][0]);work=0.
        count=max(1,int(round(duration/physics_dt)));dt=duration/count
        for step in range(count):
            d=r.quantities(q,v);contact=self.contact.solve(q,v,u,scan,d,self.force_correction,step*dt)
            work+=dt*float((v*(u@r.B.T+contact['port'])).sum()-d['dissipation'][0])
            acceleration=torch.linalg.solve(d['M'],(u@r.B.T+contact['port']-d['c'])[:,:,None]).squeeze(-1)
            v=v+dt*acceleration;q=q+dt*v
        q_error=q[0]-r.t(after['raw'][:9]);v_error=v[0]-r.t(after['raw'][9:18])
        result=dict(position_prediction_l2=float(torch.linalg.vector_norm(q_error)),velocity_prediction_l2=float(torch.linalg.vector_norm(v_error)),
            prediction_energy_balance_defect=float(r.quantities(q,v)['H'][0])-initial-work)
        if not all(np.isfinite(x) for x in result.values()):raise FloatingPointError('Non-finite held-control model prediction')
        return result
