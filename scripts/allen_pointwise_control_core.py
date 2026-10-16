"""Same-node pointwise PDE control for the fixed Allen interval comparison.

The pointwise objective averages squared residuals, not an averaged residual's
square. Nodes, quadrature weights, spatial band, physical scaling and query RNG
match the existing joint interval loss. No reliability map or solver is used.
"""
import torch
import scripts.physics_stress_core as engine
from scripts.periodic_curvature_core import make_model
from scripts.allen_interval_pilot_core import interval_windows
from scripts.allen_interval_replication_core import replication_penalties
from src.physics_equations_v1 import equation_residual
from src.time_fno import coefficient_energy, time_fields


def pointwise_node_penalty(model, x, t, coords, scale, offset, c):
    center, h, _ = interval_windows(t, c)
    nodes = torch.linspace(-1, 1, 5, dtype=t.dtype, device=t.device)
    queries = center[:, None] + h[:, None] * nodes
    u, ut = time_fields(model, x.repeat_interleave(5, dim=0), queries.reshape(-1),
                        coords, scale, offset.repeat_interleave(5), create_graph=True)
    residual = equation_residual(u, ut, c) * (c['time_upper'] / scale)
    energies = coefficient_energy(residual).reshape(len(x), 5)
    quadrature = torch.tensor([1, 4, 2, 4, 1], dtype=u.dtype, device=u.device) / 12
    return (energies * quadrature[None, :]).sum(1).mean()


def corecontrol_penalties(model, x, t, coords, scale, offset, c):
    penalties = replication_penalties(model, x, t, coords, scale, offset, c)
    penalties['pointwise_nodes'] = pointwise_node_penalty(model, x, t, coords, scale, offset, c)
    return penalties


def train_one(c, a, context, folder, device, warm_path=None, stop_after=None):
    """Sequential-only adapter; restore the sealed engine globals on all exits."""
    factory, penalty = engine.TimeFNO1d, engine.penalties
    try:
        engine.TimeFNO1d = lambda width, modes, layers: make_model(c, width, modes, layers)
        engine.penalties = corecontrol_penalties
        return engine.train_one(c, a, context, folder, device, warm_path, stop_after)
    finally:
        engine.TimeFNO1d, engine.penalties = factory, penalty
