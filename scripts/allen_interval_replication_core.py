"""Observation-only Allen replication adapter with no reliability-map input.

The full-curvature and uniform joint-interval operations are delegated unchanged
to their sealed implementations. Every continuation computes both forwards;
only its selected objective enters backward. The shared engine is sequential
only, and its model/penalty globals are restored even after interruption/error.
"""
import scripts.physics_stress_core as engine
from scripts.periodic_curvature_core import make_model, curvature_penalties
from scripts.allen_interval_pilot_core import joint_interval_residual
from src.time_fno import coefficient_energy


def replication_penalties(model, x, t, coords, scale, offset, c):
    full = curvature_penalties(model, x, t, coords, scale, offset, c)['full']
    residual = joint_interval_residual(model, x, t, coords, scale, offset, c)
    return dict(full=full, interval_uniform=coefficient_energy(residual).mean())


def train_one(c, a, context, folder, device, warm_path=None, stop_after=None):
    """Use the unchanged full-state engine with this two-objective adapter."""
    factory, penalty = engine.TimeFNO1d, engine.penalties
    try:
        engine.TimeFNO1d = lambda width, modes, layers: make_model(c, width, modes, layers)
        engine.penalties = replication_penalties
        return engine.train_one(c, a, context, folder, device, warm_path, stop_after)
    finally:
        engine.TimeFNO1d, engine.penalties = factory, penalty
