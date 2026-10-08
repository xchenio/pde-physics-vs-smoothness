"""Parseval-closed Fourier diagnostics and independent periodic PDE terms."""
import numpy as np


def coefficients(u):return np.fft.rfft(np.asarray(u,dtype=float),axis=-1,norm='forward')


def weights(n):
    w=np.full(n//2+1,2.);w[0]=1.
    if n%2==0:w[-1]=1.
    return w


def derivative(u,order,length):
    n=u.shape[-1];k=2*np.pi*np.fft.rfftfreq(n,d=length/n)
    f=(1j*k)**order
    if order==1 and n%2==0:f[-1]=0
    return np.fft.irfft(coefficients(u)*f,n=n,axis=-1,norm='forward')


def band_errors(pred,true,initial,length,columns):
    n=true.shape[-1];pf=coefficients(pred);tf=coefficients(true);w=weights(n)
    k=2*np.pi*np.fft.rfftfreq(n,d=length/n);bands=[(0,4),(5,16),(17,64),(65,n//2)]
    result={}
    for order,name in [(0,'field'),(1,'gradient'),(2,'curvature')]:
        factor=np.ones_like(k) if order==0 else k**order
        if order==1:factor[-1]=0
        energy=np.abs((pf-tf)*factor)**2*w
        truth_energy=np.sum(np.abs(tf*factor)**2*w,axis=-1)
        base=initial if order==0 else derivative(initial,order,length)
        floor=max(1e-12,.001*np.sqrt(np.mean(base**2)))
        denominator=np.maximum(truth_energy,floor**2)
        total=np.sum(energy,axis=-1);parts={f'{lo}_{hi}':np.sum(energy[...,lo:hi+1],axis=-1) for lo,hi in bands}
        np.testing.assert_allclose(sum(parts.values()),total,rtol=1e-12,atol=1e-14)
        shares={b:float(v[:,columns].sum()/max(total[:,columns].sum(),1e-30)) for b,v in parts.items()}
        e=pred-true;de=e if order==0 else derivative(e,order,length)
        edge=np.concatenate([de[...,:4],de[...,-4:]],axis=-1)
        edge_fraction=float(np.sum(edge[:,columns]**2)/max(np.sum(de[:,columns]**2),1e-30))
        seam=e[...,0]-e[...,-1]
        result[name]=dict(relative_mean=float(np.sqrt(total/denominator)[:,columns].mean()),
                          absolute_rms=float(np.sqrt(total[:,columns].mean())),true_rms=float(np.sqrt(truth_energy[:,columns].mean())),
                          error_energy_shares=shares,boundary8_energy_fraction=edge_fraction,
                          seam_error_rms=float(np.sqrt(np.mean(seam[:,columns]**2))),denominator_floor_rms=floor,
                          denominator_floor_active_fraction=float(np.mean(truth_energy[:,columns]<floor**2)),
                          bands_normalized_squared_mean={b:float((v/denominator)[:,columns].mean()) for b,v in parts.items()})
    return result


def lowpass(u,n=256):
    source=coefficients(u);dest=source[...,:n//2+1].copy();dest[...,-1]=0
    return np.fft.irfft(dest,n=n,axis=-1,norm='forward')


def nonlinear_coefficients(u,power):
    n=u.shape[-1];oversample=2 if power==2 else 4
    cf=coefficients(u);pad=np.zeros(u.shape[:-1]+(oversample*n//2+1,),dtype=complex)
    pad[...,:cf.shape[-1]]=cf;pad[...,n//2]*=.5
    fine=np.fft.irfft(pad,n=oversample*n,axis=-1,norm='forward')
    return coefficients(fine**power)[...,:n//2+1]


def pde_terms(u,ut,c,band):
    uf=coefficients(u)[...,:band+1];a=coefficients(ut)[...,:band+1]
    k=np.arange(band+1)*2*np.pi/c['length'];diff=c['diffusion']*k*k*uf
    if c['pde']=='burgers':nonlinear=1j*k*.5*nonlinear_coefficients(u,2)[...,:band+1]
    else:nonlinear=-c['reaction']*(uf-nonlinear_coefficients(u,3)[...,:band+1])
    return a,diff,nonlinear


def resolved_energy(cf):return np.abs(cf[...,0])**2+2*np.sum(np.abs(cf[...,1:])**2,axis=-1)


def residual_summary(u,ut,c,band):
    terms=pde_terms(u,ut,c,band);r=sum(terms)
    den=np.sqrt(sum(resolved_energy(t) for t in terms)).clip(1e-12)
    relative=np.sqrt(resolved_energy(r))/den
    return dict(median_relative=float(np.median(relative)),mean_relative=float(relative.mean()),
                p90_relative=float(np.quantile(relative,.9)),absolute_rms=float(np.sqrt(resolved_energy(r).mean())))
