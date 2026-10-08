"""New synthetic nonlinear PDE trajectories; never overwrites old solver QA."""
import numpy as np
from scipy.integrate import solve_ivp
import torch
from scripts.allen_cahn_data import initial,reaction
from src.physics_equations_v1 import equation_residual
from src.time_fno import coefficient_energy


def trajectories(u,times,dt=.0005,length=2.,nu=.01,rho=5.):
    times=np.asarray(times,dtype=float);indices=np.rint(times/dt).astype(int)
    if np.any(np.diff(times)<=0) or abs(times[0])>1e-12 or np.max(np.abs(indices*dt-times))>1e-10:
        raise ValueError('sorted grid-aligned times starting at0 required')
    u=np.asarray(u,dtype=float).copy();n=u.shape[-1]
    k=2*np.pi*np.fft.rfftfreq(n,d=length/n);factor=np.exp(-nu*k*k*dt/2)
    fields=[u.copy()];lookup=set(indices[1:])
    for step in range(1,int(indices[-1])+1):
        u=np.fft.irfft(np.fft.rfft(u,axis=-1)*factor,n=n,axis=-1)
        u=reaction(u,dt,rho)
        u=np.fft.irfft(np.fft.rfft(u,axis=-1)*factor,n=n,axis=-1)
        if not np.isfinite(u).all():raise ValueError('solver nonfinite')
        if step in lookup:fields.append(u.copy())
    return np.stack(fields,axis=1)


def qa():
    ids=list(range(300000,300016));times=[0.,.05,.125,.25,.5];x=initial(ids)
    y=trajectories(x,times);coarse=trajectories(x,times,dt=.001);fine=trajectories(x,times,dt=.00025)
    high=trajectories(initial(ids,512),times,dt=.00025)[:,:,::2]
    rel=lambda a,b:np.linalg.norm(a-b,axis=-1)/np.maximum(np.linalg.norm(b,axis=-1),1e-12)
    k=2*np.pi*np.fft.rfftfreq(256,d=2/256)
    rhs=lambda t,v:.01*np.fft.irfft(-k*k*np.fft.rfft(v),n=256)+5*v*(1-v*v)
    reference=solve_ivp(rhs,(0,.5),x[0],method='DOP853',rtol=1e-10,atol=1e-12,t_eval=times).y.T
    neighbors=sorted({0.}|{t+d for t in times[1:] for d in [-.0001,0.,.0001]})
    values=trajectories(x,neighbors,dt=.0001);residuals=[]
    c=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=5.,resolved_max_mode=16)
    for t in times[1:]:
        center=int(np.argmin(np.abs(np.asarray(neighbors)-t)))
        u=torch.tensor(values[:,center],dtype=torch.float64)
        ut=torch.tensor((values[:,center+1]-values[:,center-1])/.0002,dtype=torch.float64)
        r=equation_residual(u,ut,c)
        denom=coefficient_energy(torch.fft.rfft(ut,norm='forward')[:,:17]).sqrt().clamp_min(1e-12)
        residuals.append(float((coefficient_energy(r).sqrt()/denom).max()))
    tc=rel(coarse,fine)[:,1:];tf=rel(y,fine)[:,1:]
    constant=np.full((1,256),.25)
    result=dict(qa_ids=ids,time_max_relative=float(tf.max()),space_max_relative=float(rel(fine,high).max()),
                refinement_ratio_min=float((tc/np.maximum(tf,1e-20)).min()),independent_ode_relative=float(rel(y[0],reference).max()),
                constant_max_error=float(np.abs(trajectories(constant,times)[:,-1]-reaction(constant,.5)).max()),
                nonlinear_departure=float(rel(y[:,-1],trajectories(x,times,rho=0)[:,-1]).mean()),
                resolved_relative_residual_max_by_time=residuals)
    result['passed']=bool(result['time_max_relative']<1e-4 and result['space_max_relative']<1e-5 and
                          result['refinement_ratio_min']>3 and result['independent_ode_relative']<1e-4 and
                          result['constant_max_error']<1e-10 and max(residuals)<.01)
    return result
