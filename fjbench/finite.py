"""Positive finite twin-critic operator and exact residual recovery."""
import numpy as np
from bench.operators import phase_coefficients


class ResidualRecoveryError(FloatingPointError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__("Residual recovery failed the absolute/relative gate: "
            f"defect={diagnostics['inverse_residual']:.6g}, allowed={diagnostics['inverse_allowed']:.6g}, "
            f"scale={diagnostics['inverse_scale']:.6g}, contraction_gap={diagnostics['contraction_gap']:.6g}. "
            "Inspect value/reward scale and operator conditioning.")


def entropy(h, config):
    u = np.asarray(h)/config["reference_time"]
    if config["entropy_convention"] == "start":
        return config["alpha_reference"]*u
    beta = -np.log(config["gamma_reference"])
    return config["alpha_reference"]*(-np.expm1(-beta*u))/beta


def validate_rows(p, gamma):
    if not np.isfinite(p).all() or p.ndim != 3 or np.any(p < 0) or not np.allclose(p.sum(-1),1.,atol=1e-10,rtol=0):
        raise ValueError("Each state/action row must be a positive probability vector")
    if not np.isfinite(gamma).all() or np.any(gamma < 0) or np.any(gamma >= 1):
        raise ValueError("The frozen operator requires discounts in [0,1)")


def backup(q, rewards, gamma, p, entropy_logp):
    # q: [2,n,k]. Actor quadrature is uniform over k frozen reparameterized samples.
    soft_v = (np.minimum(q[0],q[1])-entropy_logp).mean(-1)
    base = rewards + gamma*np.einsum("nkl,l->nk",p,soft_v)
    return np.stack([base,base])


def recover(error, rewards, gamma, p, entropy_logp,refinement_steps=0,return_stats=False):
    """Solve q-U(q)=error while both candidate heads remain free variables.

    q_j=R+Gamma P v+error_j. Thus min(q_1,q_2) adds min(error_1,error_2)
    to their common base. Only one n-by-n linear solve is needed.
    """
    a = np.eye(len(p))-(gamma[...,None]*p).mean(1)
    rhs = (rewards+np.minimum(error[0],error[1])-entropy_logp).mean(1)
    value = np.linalg.solve(a,rhs)
    solves=1
    # Evaluate the linear residual at extended precision where supported.
    # The returned critic values still use float64.
    for _ in range(refinement_steps):
        residual=np.asarray(rhs.astype(np.longdouble)-a.astype(np.longdouble)@value.astype(np.longdouble),dtype=float)
        if not np.isfinite(residual).all():break
        correction=np.linalg.solve(a,residual);solves+=1
        candidate=value+correction
        after=np.asarray(rhs.astype(np.longdouble)-a.astype(np.longdouble)@candidate.astype(np.longdouble),dtype=float)
        if np.max(abs(after))>=np.max(abs(residual)):break
        value=candidate
    common = rewards+gamma*np.einsum("nkl,l->nk",p,value)
    result=common[None]+error
    return (result,solves) if return_stats else result


def residual_vjp(q, rho, gamma, p):
    active = q[0] <= q[1]
    shared = np.einsum("nk,nkl->l",(rho[0]+rho[1])*gamma,p)/q.shape[-1]
    return rho - np.stack([shared[:,None]*active,shared[:,None]*(~active)])


def solve(q, rewards, gamma, p, entropy_logp, config, method="ph_sac"):
    validate_rows(p,gamma)
    if any(not np.isfinite(x).all() for x in (q,rewards,entropy_logp)):raise FloatingPointError("Non-finite finite-operator input")
    q=np.array(q,dtype=float,copy=True)
    rho=np.zeros_like(q)
    calls=0
    error=q-backup(q,rewards,gamma,p,entropy_logp);calls+=1
    initial=.5*np.sum(error**2)
    largest_inverse,largest_balance=0.,0.
    recovery_port=0.;phase_count=0
    inverse_allowed=inverse_scale=inverse_relative=0.;inverse_gate=True
    gap=float(1-np.max(gamma))
    def check_inverse(candidate,measured,requested):
        defect=float(np.max(abs(measured-requested)))
        scale=float(max(1.,np.max(abs(candidate)),np.max(abs(candidate-measured)),np.max(abs(requested))))
        allowed=config['inverse_tolerance']+config.get('inverse_relative_tolerance',1e-12)*scale
        result=dict(inverse_residual=defect,inverse_allowed=allowed,inverse_scale=scale,
            inverse_absolute_gate_pass=bool(defect<=config['inverse_tolerance']),
            inverse_relative_residual=defect/scale,inverse_gate_pass=bool(np.isfinite(defect) and defect<=allowed),
            contraction_gap=gap,recovery_condition_bound=(2-gap)/gap,solver_recovery_revision=2,
            operator_calls=calls,linear_solves=solves,phase_steps=phase_count)
        if not result['inverse_gate_pass']:raise ResidualRecoveryError(result)
        return result
    if method == "ph_direct":
        q,solves=recover(np.zeros_like(q),rewards,gamma,p,entropy_logp,config.get('inverse_refinement_steps',2),True)
        error=q-backup(q,rewards,gamma,p,entropy_logp);calls+=1
        checked=check_inverse(q,error,np.zeros_like(q))
        largest_inverse=checked['inverse_residual'];inverse_allowed=checked['inverse_allowed'];inverse_scale=checked['inverse_scale'];inverse_relative=checked['inverse_relative_residual']
    else:
        solves=0
        for _ in range(config["phase_steps"]):
            phase_count+=1
            if method == "interval_euler":
                q=q-config["euler_eta"]*error
            else:
                before=.5*np.sum(error**2+rho**2)
                c,s=phase_coefficients(config["sigma"],config["damping"],config["omega"])
                enew,rnew=c*error+s*rho,-s*error+c*rho
                midpoint_e,midpoint_r=(error+enew)/2,(rho+rnew)/2
                after=.5*np.sum(enew**2+rnew**2)
                balance=after-before+config["sigma"]*config["damping"]*np.sum(midpoint_e**2+midpoint_r**2)
                largest_balance=max(largest_balance,abs(float(balance)))
                q,used=recover(enew,rewards,gamma,p,entropy_logp,config.get('inverse_refinement_steps',2),True);solves+=used
                measured=q-backup(q,rewards,gamma,p,entropy_logp);calls+=1
                recovery_port+=float(.5*np.sum(measured**2+rnew**2)-after)
                largest_inverse=max(largest_inverse,float(np.max(abs(measured-enew))))
                checked=check_inverse(q,measured,enew)
                inverse_allowed=max(inverse_allowed,checked['inverse_allowed']);inverse_scale=max(inverse_scale,checked['inverse_scale'])
                inverse_relative=max(inverse_relative,checked['inverse_relative_residual'])
                rho=rnew
            error=q-backup(q,rewards,gamma,p,entropy_logp);calls+=1
            if np.max(abs(error))<=config["residual_tolerance"] and np.max(abs(rho))<=config["momentum_tolerance"]:break
    final=.5*np.sum(error**2+rho**2)
    return q,rho,dict(energy_before=float(initial),energy_after=float(final),
        inverse_residual=largest_inverse,midpoint_balance_defect=largest_balance,
        inverse_allowed=inverse_allowed,inverse_scale=inverse_scale,inverse_relative_residual=inverse_relative,
        inverse_gate_pass=inverse_gate,contraction_gap=gap,recovery_condition_bound=(2-gap)/gap,solver_recovery_revision=2,
        inverse_absolute_gate_pass=bool(largest_inverse<=config['inverse_tolerance']),
        residual_inf=float(np.max(abs(error))),contraction=float(np.max(gamma)),
        operator_calls=calls,linear_solves=solves,
        phase_steps=phase_count,recovery_storage_port=recovery_port,
        solver_converged=bool(np.max(abs(error))<=config["residual_tolerance"] and np.max(abs(rho))<=config["momentum_tolerance"]),
        cotangent_norm=float(np.linalg.norm(residual_vjp(q,rho,gamma,p))),
        finite_value_bound=float(np.max(abs(error))/(1-np.max(gamma))))
