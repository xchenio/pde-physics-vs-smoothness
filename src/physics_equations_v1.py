"""Resolved periodic equations for fixed-architecture stress experiments."""
import torch
from src.time_fno import conservative_residual, coefficient_energy, time_fields


def dealiased_cube(u):
    n=u.shape[-1]
    if n%2:raise ValueError('even spatial grid required')
    coefficients=torch.fft.rfft(u,norm='forward')
    padded=torch.nn.functional.pad(coefficients,(0,3*n//2))
    mask=torch.ones_like(padded.real);mask[...,n//2]=.5
    fine=torch.fft.irfft(padded*mask,n=4*n,norm='forward')
    return torch.fft.rfft(fine**3,norm='forward')[...,:n//2+1]


def equation_residual(u,ut,c):
    band=c['resolved_max_mode'];length=c['length'];nu=c['diffusion']
    if c['pde']=='burgers':return conservative_residual(u,ut,nu,band,length)
    if c['pde']!='allen_cahn':raise ValueError('unknown PDE')
    k=torch.arange(band+1,device=u.device,dtype=u.dtype)*2*torch.pi/length
    uf=torch.fft.rfft(u,norm='forward')[...,:band+1]
    utf=torch.fft.rfft(ut,norm='forward')[...,:band+1]
    return utf+nu*k.square()*uf-c['reaction']*(uf-dealiased_cube(u)[...,:band+1])


def penalties(model,x,t,coords,scale,offset,c,create_graph=True):
    u,ut=time_fields(model,x,t,coords,scale,offset,create_graph=create_graph)
    band=c['resolved_max_mode'];k=torch.arange(band+1,device=u.device,dtype=u.dtype)*2*torch.pi/c['length']
    uf=torch.fft.rfft(u,norm='forward')[...,:band+1]
    return dict(physics=coefficient_energy(equation_residual(u,ut,c)).mean()/scale**2,
                curvature=coefficient_energy(-k.square()*uf).mean()/scale**2)
