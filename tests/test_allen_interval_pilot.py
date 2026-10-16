import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.allen_interval_pilot_core import (
    interval_penalties, interval_windows, joint_interval_residual,
    reliability_weights, train_one,
)
from scripts.periodic_curvature_core import curvature_penalties
from src.periodic_position_fno import PositionTimeFNO1d
from src.resume import random_state


def config(ones=False):
    centers = [0., .1, .2, .35, .5]
    values = np.ones((5, 2, 17)) if ones else (
        .15 + .025 * np.arange(5)[:, None, None] +
        .05 * np.arange(2)[None, :, None] +
        .005 * np.arange(17)[None, None, :])
    return dict(
        pde='allen_cahn', length=2., diffusion=.01, reaction=5.,
        resolved_max_mode=16, interval_half_widths=[.005, .01],
        time_lower=.05, time_upper=.5, interval_nodes=5,
        reliability_map=dict(centers=centers, half_widths=[.005, .01],
                             weights32=values.tolist()))


class ScalarTimeField(torch.nn.Module):
    """Spatially constant field with an optional trainable affine time path."""
    def __init__(self, bias=.2, slope=0.):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float64))
        self.slope = torch.nn.Parameter(torch.tensor(slope, dtype=torch.float64))

    def forward(self, x, t, coords):
        return (self.bias + self.slope * t[:, None]).expand_as(x)


class AllenIntervalPilotTests(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertIsInstance(second, torch.Tensor)
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            self.assertTrue(np.array_equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_reliability_interpolation_positive_and_edge_clamped(self):
        c = config()
        query = [-.1, 0., .05, .1, .125, .2, .275, .35, .5, .7]
        wi = np.arange(len(query)) % 2
        table = np.asarray(c['reliability_map']['weights32'])
        expected = np.stack([
            [np.interp(t, c['reliability_map']['centers'], table[:, width, mode])
             for mode in range(17)] for t, width in zip(query, wi)])
        for device in self.devices():
            with self.subTest(device=device):
                t = torch.tensor(query, device=device, dtype=torch.float64)
                indices = torch.tensor(wi, device=device)
                got = reliability_weights(t, indices, c)
                self.assertEqual(got.shape, (len(query), 17))
                self.assertEqual(got.dtype, t.dtype)
                self.assertTrue(bool(torch.all((got > 0) & (got <= 1))))
                torch.testing.assert_close(got, torch.tensor(expected, device=device),
                                           rtol=1e-14, atol=1e-14)
                bad = config()
                bad['reliability_map']['weights32'][0][0][0] = 0.
                with self.assertRaises(ValueError):
                    reliability_weights(t, indices, bad)

    def test_windows_cover_domain_with_alternating_widths_without_rng(self):
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                t = torch.tensor([.05, .05, .275, .275, .5, .5],
                                 dtype=torch.float64, device=device)
                generator = np.random.default_rng(108)
                before = random_state(generator)
                center, h, wi = interval_windows(t, c)
                after = random_state(generator)
                self.assert_tree_equal(before, after)
                self.assertEqual(wi.tolist(), [0, 1, 0, 1, 0, 1])
                expected_h = torch.tensor([.005, .01] * 3, dtype=t.dtype, device=device)
                expected_center = torch.tensor([.005, .01, .25, .25, .495, .49],
                                               dtype=t.dtype, device=device)
                torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
                torch.testing.assert_close(center, expected_center, rtol=1e-14, atol=1e-14)
                self.assertTrue(bool(torch.all(center - h >= -1e-14)))
                self.assertTrue(bool(torch.all(center + h <= .5 + 1e-14)))

    def test_constant_physical_joint_rate_matches_analytic_allen_rhs(self):
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                model = ScalarTimeField(bias=.17).to(device)
                x = torch.zeros(4, 64, dtype=torch.float64, device=device)
                t = torch.tensor([.05, .16, .3, .5], dtype=x.dtype, device=device)
                coords = torch.arange(64, dtype=x.dtype, device=device) / 32
                offset = torch.tensor([-.1, 0., .1, .2], dtype=x.dtype, device=device)
                scale = 1.7
                got = joint_interval_residual(model, x, t, coords, scale, offset, c)
                self.assertEqual(got.shape, (4, 17))
                self.assertTrue(got.is_complex())
                u = scale * model.bias + offset
                expected = -c['reaction'] * u * (1 - u.square()) * c['time_upper'] / scale
                torch.testing.assert_close(got[:, 0].real, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(got[:, 0].imag, torch.zeros_like(expected), rtol=0, atol=0)
                torch.testing.assert_close(got[:, 1:], torch.zeros_like(got[:, 1:]), rtol=0, atol=1e-12)

    def test_joint_affine_path_keeps_endpoint_and_rhs_parameter_gradients(self):
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                model = ScalarTimeField(bias=.13, slope=.31).to(device)
                x = torch.zeros(4, 64, dtype=torch.float64, device=device)
                t = torch.tensor([.05, .16, .3, .5], dtype=x.dtype, device=device)
                coords = torch.arange(64, dtype=x.dtype, device=device) / 32
                offset = torch.tensor([-.1, 0., .1, .2], dtype=x.dtype, device=device)
                scale = 1.7
                center, h, _ = interval_windows(t, c)
                u = scale * (model.bias + model.slope * center) + offset
                velocity = scale * model.slope
                # Simpson5 integrates this cubic RHS exactly. The joint rate is
                # endpoint velocity minus the interval-averaged Allen reaction.
                expected = (velocity - c['reaction'] * (
                    u - u.pow(3) - u * (velocity * h).square())) * c['time_upper'] / scale
                got = joint_interval_residual(model, x, t, coords, scale, offset, c)
                torch.testing.assert_close(got[:, 0].real, expected, rtol=1e-11, atol=1e-11)
                parameters = tuple(model.parameters())
                actual_grad = torch.autograd.grad(got.abs().square().sum(), parameters,
                                                  retain_graph=True)
                expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
                for actual, analytic in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, analytic, rtol=1e-10, atol=1e-10)
                    self.assertGreater(float(abs(actual)), 0.)

    def test_ones_weights_equal_losses_gradients_and_old_curvature_no_extra_rng(self):
        torch.set_num_threads(2)
        c = config(ones=True)
        for device in self.devices():
            with self.subTest(device=device):
                torch.manual_seed(41)
                model = PositionTimeFNO1d(4, 4, 2, position='periodic', length=2.).to(device)
                torch.nn.init.normal_(model.project[-1].weight, std=.2)
                x = torch.randn(4, 64, device=device)
                t = torch.tensor([.05, .16, .3, .5], device=device)
                coords = torch.arange(64, device=device) / 32
                offset = x.mean(-1)
                generator = np.random.default_rng(227)
                before = random_state(generator)
                ps = interval_penalties(model, x, t, coords, 1.7, offset, c)
                after = random_state(generator)
                self.assert_tree_equal(before, after)
                self.assertEqual(set(ps), {'low', 'full', 'interval_uniform', 'interval_reliable'})
                old = curvature_penalties(model, x, t, coords, 1.7, offset, c)
                for kind in ('low', 'full'):
                    torch.testing.assert_close(ps[kind], old[kind], rtol=0, atol=0)
                torch.testing.assert_close(ps['interval_uniform'], ps['interval_reliable'], rtol=0, atol=0)
                parameters = tuple(model.parameters())
                uniform = torch.autograd.grad(ps['interval_uniform'], parameters,
                                               retain_graph=True, allow_unused=True)
                reliable = torch.autograd.grad(ps['interval_reliable'], parameters,
                                                allow_unused=True)
                for first, second in zip(uniform, reliable):
                    if first is None:
                        self.assertIsNone(second)
                    else:
                        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_adapter_restores_shared_engine_globals_after_failure(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        with patch.object(engine, 'train_one', side_effect=RuntimeError('deliberate test failure')):
            with self.assertRaisesRegex(RuntimeError, 'deliberate test failure'):
                train_one(config(), {}, {}, 'unused-test-folder', torch.device('cpu'))
        self.assertIs(engine.TimeFNO1d, factory)
        self.assertIs(engine.penalties, penalty)

    def test_all_five_arms_cpu_cuda_exact_resume_and_shared_query_streams(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(73)
        x = rng.normal(size=(8, 64)).astype('float32')
        arrays = dict(train_x=x, train_y=np.stack([x, .8 * x, .5 * x], 1),
                      val_x=x[:2], val_y=np.stack([x[:2], .8 * x[:2], .5 * x[:2]], 1),
                      times=[0., .25, .5], val_times=[0., .25, .5], grid256=np.arange(64) / 32)
        base = dict(config(), seed=0, steps=4, width=4, modes=4, layers=2,
                    lr=.001, lr_min=.00001, weight_decay=.0001, batch_size=4,
                    eval_every=2, checkpoint_every=2, evaluation_columns=[1, 2],
                    position='periodic', position_center=1., position_scale=.8)
        for device in self.devices():
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                train_one(dict(base, kind='warm', strength=0.), arrays, {}, root / 'warm',
                          torch.device(device))
                warm_path = root / 'warm/checkpoints/latest.pt'
                paired = []
                for kind in ('none', 'low', 'full', 'interval_uniform', 'interval_reliable'):
                    c = dict(base, kind=kind, strength=0. if kind == 'none' else .3, lr=.0003)
                    first = root / (kind + '_full')
                    second = root / (kind + '_resumed')
                    complete = train_one(c, arrays, {}, first, torch.device(device), warm_path=warm_path)
                    partial = train_one(c, arrays, {}, second, torch.device(device), warm_path=warm_path,
                                        stop_after=2)
                    self.assertIsNone(partial)
                    resumed = train_one(c, arrays, {}, second, torch.device(device), warm_path=warm_path)
                    for key in ('history', 'initial', 'batch_hash', 'collocation_hash',
                                'calibration', 'penalty_weight'):
                        self.assert_tree_equal(complete[key], resumed[key])
                    ca = torch.load(first / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                    cb = torch.load(second / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                    for key in ca:
                        if key != 'seconds':
                            self.assert_tree_equal(ca[key], cb[key])
                    if kind != 'none':
                        self.assertAlmostEqual(complete['history'][0]['effective_gradient_ratio'], .3,
                                               places=6)
                    paired.append((complete['initial'], complete['batch_hash'],
                                   complete['collocation_hash'], complete['calibration']))
                    self.assertIs(engine.TimeFNO1d, factory)
                    self.assertIs(engine.penalties, penalty)
                self.assertTrue(all(value == paired[0] for value in paired))


if __name__ == '__main__':
    unittest.main()
