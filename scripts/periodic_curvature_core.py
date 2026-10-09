"""Versioned adapter to sealed resumable stress core, no added modules.

The adapter replaces ONLY the model factory and penalty dictionary for one
sequential call and restores both globals even on failure. Provenance includes
this adapter and the unchanged delegated core. Not thread-safe; one writer only.
"""
import torch
from src.periodic_position_fno import PositionTimeFNO1d
from src.time_fno import time_fields,coefficient_energy
import scripts.physics_stress_core as engine


def curvature_penalties(model,x,t,coords,scale,offset,c):
    u,_=time_fields(model,x,t,coords,scale,offset,create_graph=True)
    n=u.shape[-1];uf=torch.fft.rfft(u,norm='forward')
    k=torch.arange(n//2+1,device=u.device,dtype=u.dtype)*2*torch.pi/c['length']
    cf=-k.square()*uf;band=c['resolved_max_mode']
    lowk=torch.arange(band+1,device=u.device,dtype=u.dtype)*2*torch.pi/c['length']
    # Preserve the historical low-band tensor operations, including slice layout.
    low=coefficient_energy(-lowk.square()*uf[...,:band+1]).mean()/scale**2
    values=cf.abs().square()
    # Single real DC/Nyquist, double positive-frequency conjugate pairs.
    full=(values[...,0]+values[...,-1]+2*values[...,1:-1].sum(-1)).mean()/scale**2
    return dict(low=low,full=full)


def make_model(c,width,modes,layers):
    return PositionTimeFNO1d(width,modes,layers,position=c['position'],length=c['length'],
                             position_center=c['position_center'],position_scale=c['position_scale'])


def train_one(c,a,context,folder,device,warm_path=None,stop_after=None):
    factory,penalty=engine.TimeFNO1d,engine.penalties
    try:
        engine.TimeFNO1d=lambda width,modes,layers:make_model(c,width,modes,layers)
        engine.penalties=curvature_penalties
        return engine.train_one(c,a,context,folder,device,warm_path,stop_after)
    finally:
        engine.TimeFNO1d,engine.penalties=factory,penalty
