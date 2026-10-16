"""Exploratory Allen-only joint interval adapter; no new trainable modules.

All continuations compute the same penalty forwards. Only the chosen penalty
enters backward, so equal steps do NOT imply equal FLOPs. The sealed curvature
operations and query RNG are preserved for historical tensor replay.
"""
import torch
import scripts.physics_stress_core as engine
from scripts.periodic_curvature_core import make_model, curvature_penalties
from src.physics_equations_v1 import equation_residual
from src.time_fno import coefficient_energy


def reliability_weights(t_centers, width_indices, c):
    """Positive fixed training-only weights, linearly interpolated in time."""
    m = c['reliability_map']
    centers = torch.as_tensor(m['centers'], dtype=t_centers.dtype, device=t_centers.device)
    weights = torch.as_tensor(m['weights32'], dtype=t_centers.dtype, device=t_centers.device)
    if weights.shape != (len(centers), len(c['interval_half_widths']), c['resolved_max_mode'] + 1):
        raise ValueError('reliability map shape mismatch')
    if not torch.all(torch.isfinite(weights) & (weights > 0) & (weights <= 1)):
        raise ValueError('invalid continuous weights')
    t = t_centers.clamp(centers[0], centers[-1])
    right = torch.searchsorted(centers, t.contiguous(), right=True).clamp(1, len(centers) - 1)
    left = right - 1
    fraction = ((t - centers[left]) / (centers[right] - centers[left]))[:, None]
    return (1 - fraction) * weights[left, width_indices] + fraction * weights[right, width_indices]


def interval_windows(t, c):
    """Derive balanced widths/valid centers without consuming any new RNG."""
    if c['interval_nodes'] != 5 or c['pde'] != 'allen_cahn':
        raise ValueError('pilot is Allen-Cahn with Simpson5 only')
    widths = torch.as_tensor(c['interval_half_widths'], dtype=t.dtype, device=t.device)
    wi = torch.arange(len(t), device=t.device) % len(widths)
    h = widths[wi]
    total = c['time_upper']
    fraction = (t - c['time_lower']) / (c['time_upper'] - c['time_lower'])
    center = h + (total - 2 * h) * fraction
    return center, h, wi


def joint_interval_residual(model, x, t, coords, scale, offset, c):
    """Resolved Fourier RATE residual, dimensionless by training scale / T.

    Both endpoint and RHS terms depend on this same trainable model. No frozen
    future field, intermediate reference label, or AD time target is supplied.
    """
    center, h, _ = interval_windows(t, c)
    nodes = torch.linspace(-1, 1, 5, dtype=t.dtype, device=t.device)
    query = center[:, None] + h[:, None] * nodes
    u = model(x.repeat_interleave(5, dim=0), query.reshape(-1), coords)
    u = u.reshape(len(x), 5, -1) * scale + offset[:, None, None]
    rhs = -equation_residual(u, torch.zeros_like(u), c)
    quadrature = torch.tensor([1, 4, 2, 4, 1], dtype=u.dtype, device=u.device) / 12
    average_rhs = (rhs * quadrature[None, :, None]).sum(1)
    endpoint_rate = torch.fft.rfft((u[:, -1] - u[:, 0]) / (2 * h[:, None]), norm='forward')
    endpoint_rate = endpoint_rate[..., :c['resolved_max_mode'] + 1]
    return (endpoint_rate - average_rhs) * (c['time_upper'] / scale)


def interval_penalties(model, x, t, coords, scale, offset, c):
    penalties = curvature_penalties(model, x, t, coords, scale, offset, c)
    residual = joint_interval_residual(model, x, t, coords, scale, offset, c)
    center, _, wi = interval_windows(t, c)
    weights = reliability_weights(center, wi, c)
    penalties['interval_uniform'] = coefficient_energy(residual).mean()
    energy = residual.abs().square() * weights
    penalties['interval_reliable'] = (energy[:, 0] + 2 * energy[:, 1:].sum(-1)).mean()
    return penalties


def train_one(c, a, context, folder, device, warm_path=None, stop_after=None):
    """Sequential-only adapter; restore shared engine globals on any exit."""
    factory, penalty = engine.TimeFNO1d, engine.penalties
    try:
        engine.TimeFNO1d = lambda width, modes, layers: make_model(c, width, modes, layers)
        engine.penalties = interval_penalties
        return engine.train_one(c, a, context, folder, device, warm_path, stop_after)
    finally:
        engine.TimeFNO1d, engine.penalties = factory, penalty
