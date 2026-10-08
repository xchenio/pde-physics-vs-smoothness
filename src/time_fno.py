"""Continuous-time FNO and resolved periodic conservative Burgers diagnostics.

New versioned implementation; historical discrete-head FNO is untouched.
Spatial derivatives are spectral, not derivatives with respect to sensor inputs.
"""
import torch
from torch import nn
from src.fno import FNO1d


class TimeFNO1d(FNO1d):
    def __init__(self, width=16, modes=16, layers=4):
        super().__init__(width, modes, layers)
        self.lift = nn.Conv1d(3, width, 1)
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, u0, times, coordinates):
        if u0.ndim != 2 or times.shape != (u0.shape[0],):
            raise ValueError('expected u0=(batch,grid), times=(batch,)')
        x = torch.as_tensor(coordinates, device=u0.device, dtype=u0.dtype).flatten()
        if x.numel() != u0.shape[-1]:
            raise ValueError('coordinate length mismatch')
        h = self.lift(torch.stack([u0, x.expand_as(u0), times[:, None].expand_as(u0)], 1))
        for spec, local in zip(self.spectral, self.local):
            h = torch.nn.functional.gelu(spec(h) + local(h))
        return self.project(h).squeeze(1)


def time_fields(model, u0, times, coordinates, scale, offset, create_graph=True):
    """Derivative in query time, holding input trajectory and grid fixed."""
    def call(t):
        return model(u0, t, coordinates) * scale + offset[:, None]
    return torch.autograd.functional.jvp(call, times, torch.ones_like(times),
                                         create_graph=create_graph, strict=False)


def spectral_derivative(u, order, length=1.):
    if order not in (1, 2):
        raise ValueError('only first and second spatial derivatives supported')
    n = u.shape[-1]
    k = 2 * torch.pi * torch.fft.rfftfreq(n, d=length/n, device=u.device, dtype=u.dtype)
    factor = (1j * k) if order == 1 else -k.square()
    if order == 1 and n % 2 == 0:
        factor = factor.clone(); factor[-1] = 0  # real grid Nyquist sine is undefined
    return torch.fft.irfft(torch.fft.rfft(u) * factor, n=n)


def dealiased_flux(u):
    """Exact low-frequency coefficients of interpolated u^2/2, normalized FFT.

    2x oversampling avoids product aliasing in the retained original spectrum.
    Split the real even-grid Nyquist coefficient when embedding it in a finer grid.
    """
    n = u.shape[-1]
    if n % 2:
        raise ValueError('even spatial grid required')
    coefficients = torch.fft.rfft(u, norm='forward')
    pad = torch.nn.functional.pad(coefficients, (0, n//2))
    mask = torch.ones_like(pad.real); mask[..., n//2] = .5
    fine = torch.fft.irfft(pad * mask, n=2*n, norm='forward')
    return torch.fft.rfft(.5 * fine.square(), norm='forward')[..., :n//2+1]


def conservative_residual(u, ut, diffusion, max_mode=16, length=1.):
    """Resolved weak residual coefficients, modes0..max_mode inclusive.

    This is a projected constraint, NOT the full high-frequency PDE residual.
    """
    n = u.shape[-1]
    if u.shape != ut.shape or not 0 <= max_mode < n//2:
        raise ValueError('field/time derivative shape or resolved band invalid')
    k = torch.arange(max_mode+1, device=u.device, dtype=u.dtype) * (2*torch.pi/length)
    uf = torch.fft.rfft(u, norm='forward')[..., :max_mode+1]
    utf = torch.fft.rfft(ut, norm='forward')[..., :max_mode+1]
    flux = dealiased_flux(u)[..., :max_mode+1]
    return utf + 1j*k*flux + diffusion*k.square()*uf


def coefficient_energy(coefficients):
    """Parseval mean-square of a low-mode real field, without a Nyquist mode."""
    values = coefficients.abs().square()
    return values[..., 0] + 2*values[..., 1:].sum(-1)
