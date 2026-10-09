import unittest
import numpy as np
from scripts.interval_diagnostics_core import simpson,indices,interval_metrics


class IntervalDiagnosticTests(unittest.TestCase):
    def test_simpson_complex_cubic_and_invalid_nodes(self):
        t=np.linspace(.1,.3,17);v=(t**3+1j*t)[None,:,None]
        np.testing.assert_allclose(simpson(v,t),[[(.3**4-.1**4)/4+1j*(.3**2-.1**2)/2]],rtol=1e-13)
        with self.assertRaises(ValueError):simpson(v[:,:4],t[:4])
        with self.assertRaises(ValueError):simpson(v[:,:3],[.1,.12,.15])
        np.testing.assert_array_equal(indices(t,[t[0],t[8],t[-1]]),[0,8,16])
        with self.assertRaises(ValueError):indices(t,[.11])

    def test_exact_diffusion_increment_identity_and_repair(self):
        t=np.linspace(.09,.11,17);x=np.arange(1024)/1024;nu=.01;k=2*np.pi
        truth=np.exp(-nu*k*k*t)[None,:,None]*np.sin(k*x)[None,None,:]
        ref=dict(times=t,fields={1024:truth,512:truth[...,::2]},coarse_fields={1024:truth})
        c=dict(pde='allen_cahn',length=1.,diffusion=nu,reaction=0.)
        window=dict(center=.1,half_width=.01,model_times=t,reference_times=t)
        exact=interval_metrics(truth[...,::4],t,ref,c,window)
        self.assertLess(exact['increment']['mean'],1e-13)
        self.assertLess(exact['reference_uncertainty_max'],1e-10)
        self.assertFalse(exact['descriptive_gate'])
        pred=truth[...,::4].copy();pred+=.1*np.linspace(0,1,17)[None,:,None]
        bad=interval_metrics(pred,t,ref,c,window)
        self.assertGreater(bad['increment']['mean'],1.)
        self.assertLess(bad['fixed_integral_repair_field']['mean'],bad['endpoint_field']['mean'])
        self.assertAlmostEqual(bad['time_increment_common_energy']+bad['state_rhs_integral_common_energy']+bad['cross_common_energy'],bad['error_discrepancy_common_energy'],places=9)
        self.assertAlmostEqual(bad['integral_rhs_error']['normalized_squared_mean'],bad['state_rhs_integral_common_energy'],places=9)

    def test_zero_increment_floors_block_gate(self):
        t=np.linspace(.1,.2,5);truth=np.ones((2,5,1024))*.1
        ref=dict(times=t,fields={1024:truth},coarse_fields={})
        c=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=0.)
        r=interval_metrics(truth[...,::4],t,ref,c,dict(center=.15,half_width=.05,model_times=t,reference_times=t))
        self.assertEqual(r['increment']['floor_active_fraction'],1.)
        self.assertFalse(r['descriptive_gate'])


if __name__=='__main__':unittest.main()
