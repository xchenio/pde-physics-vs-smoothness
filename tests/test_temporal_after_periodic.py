import unittest
import numpy as np
import torch
from scripts.temporal_after_periodic_core import score_coefficients,temporal_score,decomposition,aggregate
from scripts.field_derivative_diagnostics import coefficients
from src.periodic_position_fno import PositionTimeFNO1d
from src.time_fno import time_fields


class TemporalAfterPeriodicTests(unittest.TestCase):
    def test_native_projection_avoids_aliasing_and_nyquist(self):
        x=np.arange(1024)/1024
        native=(np.sin(2*np.pi*3*x)+.3*np.cos(2*np.pi*253*x))[None,None,:]
        estimate=np.sin(2*np.pi*3*np.arange(256)/256)[None,None,:]
        r=score_coefficients(coefficients(estimate)[...,:17],coefficients(native)[...,:17],np.r_[1.,np.full(16,2.)])
        self.assertLess(r['mean'],1e-13)
        self.assertGreater(temporal_score(estimate,native[...,::4],16)['mean'],.2)
        nyquist=(-1.)**np.arange(256)
        self.assertAlmostEqual(temporal_score(2*nyquist[None,None,:],nyquist[None,None,:])['mean'],1.)

    def test_decomposition_oracles_and_cross_identity(self):
        x=np.arange(1024)*2/1024;u=.1*np.sin(np.pi*x)[None,None,:]
        c=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=5.)
        rhs=-.01*np.pi**2*u+5*u*(1-u*u)
        ref=dict(fields={1024:u},ut={(1024,.0005):rhs,(1024,.001):rhs})
        pred=u[...,::4];ut=rhs[...,::4]
        exact=decomposition(pred,ut,ref,c)
        self.assertLess(exact['predicted_residual_rms'],1e-14)
        time_bad=decomposition(pred,ut+.2,ref,c)
        self.assertAlmostEqual(time_bad['predicted_residual_rms'],.2,places=12)
        self.assertLess(time_bad['oracle_time_residual_rms'],1e-14)
        self.assertGreater(time_bad['oracle_field_rhs_residual_rms'],.19)
        both=decomposition(pred+.03,ut+.2,ref,c)
        self.assertAlmostEqual(both['discrepancy_common_normalized_squared'],
                               both['time_error_common_normalized_squared']+both['rhs_error_common_normalized_squared']+both['cross_common_normalized'],places=12)

    def test_physical_time_scale_offset_and_forward_reverse_jvp(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(3)
            model=PositionTimeFNO1d(width=4,modes=4,layers=2,position='periodic').to(device)
            torch.nn.init.normal_(model.project[-1].weight,std=.1)
            x=torch.randn(2,32,device=device);t=torch.tensor([.1,.3],device=device)
            coords=torch.arange(32,device=device)/32;off=torch.tensor([20.,-3.],device=device)
            _,a=time_fields(model,x,t,coords,2.5,off,create_graph=False)
            _,b=torch.func.jvp(lambda q:model(x,q,coords)*2.5+off[:,None],(t,),(torch.ones_like(t),))
            torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)
            # Float32 small-output FD needs a wider probe to avoid cancellation.
            h=.02
            with torch.no_grad():fd=(model(x,t+h,coords)-model(x,t-h,coords))*2.5/(2*h)
            self.assertLess(float(torch.norm(fd-a)/torch.norm(a)),.001)


if __name__=='__main__':unittest.main()
