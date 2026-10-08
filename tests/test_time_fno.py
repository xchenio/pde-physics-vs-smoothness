import unittest
import torch
from src.time_fno import (TimeFNO1d, time_fields, spectral_derivative,
                          dealiased_flux, conservative_residual, coefficient_energy)


class TimeFNOTests(unittest.TestCase):
    def test_analytic_derivatives_and_conservation(self):
        x = (torch.arange(128, dtype=torch.float64)+.5)/128
        u = torch.sin(2*torch.pi*x)[None]
        torch.testing.assert_close(spectral_derivative(u, 1), (2*torch.pi*torch.cos(2*torch.pi*x))[None])
        torch.testing.assert_close(spectral_derivative(u, 2), -(2*torch.pi)**2*u)
        diffusion = .001/torch.pi
        ut = -u*spectral_derivative(u, 1) + diffusion*spectral_derivative(u, 2)
        self.assertLess(float(coefficient_energy(conservative_residual(u, ut, diffusion)).max()), 1e-24)
        # Constant mean is a solution; nonlinear flux preserves mode0 mass.
        c = torch.full_like(u, .7)
        self.assertLess(float(coefficient_energy(conservative_residual(c, c*0, diffusion))), 1e-24)

    def test_dealiasing_and_nyquist(self):
        n = 64; x = torch.arange(n, dtype=torch.float64)/n
        u = torch.cos(2*torch.pi*25*x)[None]
        flux = dealiased_flux(u)
        self.assertLess(float(flux[:, 14].abs().max()), 1e-14)  # aliased square would contain mode14
        torch.testing.assert_close(flux[:, 0].real, torch.tensor([.25], dtype=torch.float64))
        ny = ((-1.)**torch.arange(n))[None].double()
        torch.testing.assert_close(dealiased_flux(ny)[:, 0].real, torch.tensor([.25], dtype=torch.float64))
        torch.testing.assert_close(spectral_derivative(ny, 1), torch.zeros_like(ny))

    def test_time_jvp_and_parameter_gradient_cpu_cuda(self):
        torch.set_num_threads(2)
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            torch.manual_seed(19)
            model = TimeFNO1d(width=4, modes=4, layers=2).to(device)
            torch.nn.init.normal_(model.project[-1].weight, std=.2)
            u0 = torch.randn(2, 32, device=device)
            x = (torch.arange(32, device=device)+.5)/32
            t = torch.tensor([.2, .7], device=device)
            offset = u0.mean(-1)
            u, ut = time_fields(model, u0, t, x, 2., offset)
            eps = .002
            numerical = (model(u0, t+eps, x)-model(u0, t-eps, x))/eps
            torch.testing.assert_close(ut, numerical, rtol=.015, atol=3e-5)
            loss = coefficient_energy(conservative_residual(u, ut, .001/torch.pi, 4)).mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.lift.weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
