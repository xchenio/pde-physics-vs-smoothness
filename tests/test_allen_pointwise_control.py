import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.allen_pointwise_control_core import (
    corecontrol_penalties, pointwise_node_penalty, train_one,
)
from scripts.allen_interval_pilot_core import interval_windows
from scripts.allen_interval_replication_core import replication_penalties
from src.periodic_position_fno import PositionTimeFNO1d
from src.resume import random_state


def config():
    return dict(pde='allen_cahn', length=2., diffusion=.01, reaction=5.,
                resolved_max_mode=16, interval_half_widths=[.005, .01],
                interval_nodes=5, time_lower=.05, time_upper=.5,
                position='periodic', position_center=1., position_scale=.8)


class AffineTimeField(torch.nn.Module):
    def __init__(self, bias=.13, slope=.31):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float64))
        self.slope = torch.nn.Parameter(torch.tensor(slope, dtype=torch.float64))

    def forward(self, x, t, coords):
        return (self.bias + self.slope * t[:, None]).expand_as(x)


class AllenPointwiseControlTests(unittest.TestCase):
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

    def test_constant_physical_field_analytic_residual_loss_and_gradient(self):
        c = config()
        torch.set_num_threads(2)
        for device in self.devices():
            with self.subTest(device=device):
                model = AffineTimeField(bias=.17, slope=0.).to(device)
                x = torch.zeros(4, 64, dtype=torch.float64, device=device)
                t = torch.tensor([.05, .16, .3, .5], dtype=x.dtype, device=device)
                coords = torch.arange(64, dtype=x.dtype, device=device) / 32
                offset = torch.tensor([-.1, 0., .1, .2], dtype=x.dtype, device=device)
                scale = 1.7
                actual = pointwise_node_penalty(model, x, t, coords, scale, offset, c)
                physical = scale * model.bias + offset
                rate = -c['reaction'] * physical * (1 - physical.square()) * c['time_upper'] / scale
                expected = rate.square().mean()
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                gradient = torch.autograd.grad(actual, model.bias, retain_graph=True)[0]
                analytic = torch.autograd.grad(expected, model.bias)[0]
                torch.testing.assert_close(gradient, analytic, rtol=1e-12, atol=1e-12)

    def test_affine_nonlinear_path_uses_mean_square_and_time_derivative_gradients(self):
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                model = AffineTimeField().to(device)
                x = torch.zeros(4, 64, dtype=torch.float64, device=device)
                t = torch.tensor([.05, .16, .3, .5], dtype=x.dtype, device=device)
                coords = torch.arange(64, dtype=x.dtype, device=device) / 32
                offset = torch.tensor([-.1, 0., .1, .2], dtype=x.dtype, device=device)
                scale = 1.7
                center, h, _ = interval_windows(t, c)
                query = center[:, None] + h[:, None] * torch.linspace(
                    -1, 1, 5, dtype=x.dtype, device=device)
                field = scale * (model.bias + model.slope * query) + offset[:, None]
                velocity = scale * model.slope
                residual = (velocity - c['reaction'] * (field - field.pow(3))) * c['time_upper'] / scale
                quadrature = torch.tensor([1, 4, 2, 4, 1], dtype=x.dtype, device=device) / 12
                expected = (residual.square() * quadrature[None]).sum(1).mean()
                actual = pointwise_node_penalty(model, x, t, coords, scale, offset, c)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                parameters = tuple(model.parameters())
                gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
                analytic = torch.autograd.grad(expected, parameters, retain_graph=True)
                for first, second in zip(gradients, analytic):
                    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)
                wrong_residual = (velocity.detach() - c['reaction'] * (field - field.pow(3))) * c['time_upper'] / scale
                wrong_loss = (wrong_residual.square() * quadrature[None]).sum(1).mean()
                wrong_slope = torch.autograd.grad(wrong_loss, model.slope)[0]
                self.assertGreater(float(abs(gradients[1] - wrong_slope)), 1e-8)
                # Jensen/variance identity: weighted mean squared point residual
                # is not the squared weighted mean used by an integral loss.
                mean_residual = (residual * quadrature[None]).sum(1)
                mean_square = mean_residual.square().mean()
                variance = ((residual - mean_residual[:, None]).square() * quadrature[None]).sum(1).mean()
                torch.testing.assert_close(actual, mean_square + variance, rtol=1e-12, atol=1e-12)
                self.assertGreater(float(variance.detach()), 1e-10)
                old_joint = replication_penalties(model, x, t, coords, scale, offset, c)['interval_uniform']
                # An affine time field has an exact endpoint/average-AD-ut
                # identity, so only the point-residual node variance remains.
                torch.testing.assert_close(old_joint, mean_square, rtol=1e-11, atol=1e-11)

    def test_old_full_uniform_values_gradients_and_global_rng_are_unchanged(self):
        torch.set_num_threads(2)
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                torch.manual_seed(107)
                model = PositionTimeFNO1d(4, 4, 2, position='periodic', length=2.).to(device)
                torch.nn.init.normal_(model.project[-1].weight, std=.2)
                x = torch.randn(4, 64, device=device)
                t = torch.tensor([.05, .16, .3, .5], device=device)
                coords = torch.arange(64, device=device) / 32
                offset = x.mean(-1)
                generator = np.random.default_rng(237)
                before = random_state(generator)
                with patch('scripts.allen_interval_pilot_core.reliability_weights',
                           side_effect=AssertionError('reliability map is out of scope')):
                    actual = corecontrol_penalties(model, x, t, coords, 1.7, offset, c)
                self.assert_tree_equal(before, random_state(generator))
                self.assertEqual(set(actual), {'full', 'interval_uniform', 'pointwise_nodes'})
                self.assertNotIn('reliability_map', c)
                expected = replication_penalties(model, x, t, coords, 1.7, offset, c)
                for kind in ('full', 'interval_uniform'):
                    torch.testing.assert_close(actual[kind], expected[kind], rtol=0, atol=0)
                    first = torch.autograd.grad(actual[kind], tuple(model.parameters()),
                                                retain_graph=True, allow_unused=True)
                    second = torch.autograd.grad(expected[kind], tuple(model.parameters()),
                                                 retain_graph=True, allow_unused=True)
                    for a, b in zip(first, second):
                        if a is None:
                            self.assertIsNone(b)
                        else:
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_adapter_restores_engine_globals_after_failure(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        with patch.object(engine, 'train_one', side_effect=RuntimeError('intentional failure')):
            with self.assertRaisesRegex(RuntimeError, 'intentional failure'):
                train_one(config(), {}, {}, 'unused-pointwise-test-folder', torch.device('cpu'))
        self.assertIs(engine.TimeFNO1d, factory)
        self.assertIs(engine.penalties, penalty)

    def test_dense_sparse_four_arms_exact_cpu_cuda_resume_and_observed_scaling(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(98)
        x = rng.normal(size=(8, 64)).astype('float32')
        all_y = np.stack([x, .95 * x, .85 * x, .7 * x, .5 * x], 1)
        all_times = [0., .05, .125, .25, .5]
        base = dict(config(), seed=0, steps=4, width=4, modes=4, layers=2,
                    lr=.001, lr_min=.00001, weight_decay=.0001, batch_size=4,
                    eval_every=2, checkpoint_every=2)
        for device in self.devices():
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                initializations = []
                for condition, columns, evaluation in (
                        ('dense', [0, 1, 2, 3, 4], [1, 2, 3, 4]),
                        ('sparse', [0, 4], [1, 2, 3])):
                    arrays = dict(train_x=x, train_y=all_y[:, columns], val_x=x[:2], val_y=all_y[:2],
                                  times=[all_times[i] for i in columns], val_times=all_times,
                                  grid256=np.arange(64) / 32)
                    cbase = dict(base, evaluation_columns=evaluation)
                    root = Path(directory) / condition
                    context = dict(qa='synthetic-only', condition=condition)
                    warm = train_one(dict(cbase, kind='warm', strength=0.), arrays, context,
                                     root / 'warm', torch.device(device))
                    initializations.append(warm['initial'])
                    warm_path = root / 'warm/checkpoints/latest.pt'
                    warm_cp = torch.load(warm_path, weights_only=False, map_location='cpu')
                    self.assertEqual(warm_cp['scale'], float(np.asarray(arrays['train_y'], dtype=float).std()))
                    self.assertEqual(len(warm['metrics']['train']['by_time']), len(columns))
                    self.assertEqual(len(warm['metrics']['val']['by_time']), 5)
                    paired = []
                    for kind in ('none', 'full', 'interval_uniform', 'pointwise_nodes'):
                        c = dict(cbase, kind=kind, strength=0. if kind == 'none' else .3, lr=.0003)
                        first, second = root / (kind + '_full'), root / (kind + '_resumed')
                        complete = train_one(c, arrays, context, first, torch.device(device), warm_path=warm_path)
                        self.assertIsNone(train_one(c, arrays, context, second, torch.device(device),
                                                    warm_path=warm_path, stop_after=2))
                        resumed = train_one(c, arrays, context, second, torch.device(device), warm_path=warm_path)
                        for key in ('history', 'initial', 'batch_hash', 'collocation_hash', 'calibration', 'penalty_weight'):
                            self.assert_tree_equal(complete[key], resumed[key])
                        ca = torch.load(first / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                        cb = torch.load(second / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                        for key in ca:
                            if key != 'seconds':
                                self.assert_tree_equal(ca[key], cb[key])
                        self.assertTrue(all(int(v['step']) == 8 for v in ca['optimizer']['state'].values()))
                        self.assertEqual(set(complete['calibration']['penalty_gradient_norms']),
                                         {'full', 'interval_uniform', 'pointwise_nodes'})
                        self.assertNotIn('reliability_map', ca['identity']['config'])
                        if kind != 'none':
                            self.assertAlmostEqual(complete['history'][0]['effective_gradient_ratio'], .3, places=6)
                        paired.append((complete['initial'], complete['batch_hash'],
                                       complete['collocation_hash'], complete['calibration']))
                        self.assertIs(engine.TimeFNO1d, factory)
                        self.assertIs(engine.penalties, penalty)
                    self.assertTrue(all(item == paired[0] for item in paired))
                self.assertEqual(initializations[0], initializations[1])


if __name__ == '__main__':
    unittest.main()
