import unittest
import numpy as np
from scripts.field_derivative_diagnostics import band_errors,lowpass,nonlinear_coefficients,residual_summary


class DiagnosticTests(unittest.TestCase):
    def test_parseval_derivative_band_and_lowpass(self):
        x=np.arange(256)/256;true=np.sin(2*np.pi*x)[None,None,:]
        pred=true+.001*np.cos(2*np.pi*90*x)[None,None,:]
        r=band_errors(pred,true,true[:,0],1.,[0])
        self.assertAlmostEqual(r['field']['relative_mean'],.001,places=11)
        self.assertAlmostEqual(r['curvature']['relative_mean'],8.1,places=9)
        for order in r:self.assertAlmostEqual(r[order]['error_energy_shares']['65_128'],1.,places=10)
        fine=np.sin(2*np.pi*np.arange(1024)/1024)
        np.testing.assert_allclose(lowpass(fine),true[0,0],atol=1e-14)

    def test_independent_pde_and_dealiasing(self):
        x=np.arange(256)/256;u=np.sin(2*np.pi*x)[None,:];ux=2*np.pi*np.cos(2*np.pi*x)[None,:]
        c=dict(pde='burgers',length=1.,diffusion=.001/np.pi)
        ut=-u*ux-c['diffusion']*(2*np.pi)**2*u
        self.assertLess(residual_summary(u,ut,c,16)['absolute_rms'],1e-13)
        n=64;high=np.cos(2*np.pi*25*np.arange(n)/n)[None,:]
        self.assertLess(abs(nonlinear_coefficients(high,3)[0,11]),1e-14)


if __name__=='__main__':unittest.main()
