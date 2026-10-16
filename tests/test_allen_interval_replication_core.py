import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.allen_interval_replication_core import replication_penalties, train_one
from scripts.allen_interval_pilot_core import joint_interval_residual
from scripts.periodic_curvature_core import curvature_penalties
from src.periodic_position_fno import PositionTimeFNO1d
from src.resume import random_state
from src.time_fno import coefficient_energy


def config():
    return dict(pde='allen_cahn', length=2., diffusion=.01, reaction=5.,
                resolved_max_mode=16, interval_half_widths=[.005, .01],
                interval_nodes=5, time_lower=.05, time_upper=.5,
                position='periodic', position_center=1., position_scale=.8)


class ConstantField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.17, dtype=torch.float64))

    def forward(self, x, t, coords):
        return self.bias.expand_as(x)


class AllenIntervalReplicationCoreTests(unittest.TestCase):
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

    def test_old_operations_and_gradients_exact_without_reliability_or_new_rng(self):
        torch.set_num_threads(2)
        for device in self.devices():
            with self.subTest(device=device):
                torch.manual_seed(89)
                model = PositionTimeFNO1d(4, 4, 2, position='periodic', length=2.).to(device)
                torch.nn.init.normal_(model.project[-1].weight, std=.2)
                x = torch.randn(4, 64, device=device)
                t = torch.tensor([.05, .16, .3, .5], device=device)
                coords = torch.arange(64, device=device) / 32
                offset = x.mean(-1)
                c = config()
                self.assertNotIn('reliability_map', c)
                generator = np.random.default_rng(231)
                before = random_state(generator)
                with patch('scripts.allen_interval_pilot_core.reliability_weights',
                           side_effect=AssertionError('reliability map is out of scope')):
                    actual = replication_penalties(model, x, t, coords, 1.7, offset, c)
                self.assert_tree_equal(before, random_state(generator))
                expected = dict(
                    full=curvature_penalties(model, x, t, coords, 1.7, offset, c)['full'],
                    interval_uniform=coefficient_energy(
                        joint_interval_residual(model, x, t, coords, 1.7, offset, c)).mean())
                self.assertEqual(set(actual), {'full', 'interval_uniform'})
                parameters = tuple(model.parameters())
                for kind in actual:
                    torch.testing.assert_close(actual[kind], expected[kind], rtol=0, atol=0)
                    gradients = torch.autograd.grad(actual[kind], parameters, retain_graph=True,
                                                    allow_unused=True)
                    old_gradients = torch.autograd.grad(expected[kind], parameters, retain_graph=True,
                                                        allow_unused=True)
                    for first, second in zip(gradients, old_gradients):
                        if first is None:
                            self.assertIsNone(second)
                        else:
                            torch.testing.assert_close(first, second, rtol=0, atol=0)
                self.assertEqual(sum(v.numel() * (2 if v.is_complex() else 1)
                                     for v in PositionTimeFNO1d().parameters()), 34209)

    def test_constant_physical_uniform_loss_and_gradient_are_analytic(self):
        c = config()
        for device in self.devices():
            with self.subTest(device=device):
                model = ConstantField().to(device)
                x = torch.zeros(4, 64, dtype=torch.float64, device=device)
                t = torch.tensor([.05, .16, .3, .5], dtype=x.dtype, device=device)
                coords = torch.arange(64, dtype=x.dtype, device=device) / 32
                offset = torch.tensor([-.1, 0., .1, .2], dtype=x.dtype, device=device)
                scale = 1.7
                ps = replication_penalties(model, x, t, coords, scale, offset, c)
                physical = scale * model.bias + offset
                rate = -c['reaction'] * physical * (1 - physical.square()) * c['time_upper'] / scale
                analytic = rate.square().mean()
                torch.testing.assert_close(ps['full'], torch.zeros_like(ps['full']), rtol=0, atol=1e-20)
                torch.testing.assert_close(ps['interval_uniform'], analytic, rtol=1e-12, atol=1e-12)
                gradient = torch.autograd.grad(ps['interval_uniform'], model.bias, retain_graph=True)[0]
                expected = torch.autograd.grad(analytic, model.bias)[0]
                torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)

    def test_adapter_restores_shared_globals_on_failure(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        with patch.object(engine, 'train_one', side_effect=RuntimeError('intentional failure')):
            with self.assertRaisesRegex(RuntimeError, 'intentional failure'):
                train_one(config(), {}, {}, 'unused-replication-test-folder', torch.device('cpu'))
        self.assertIs(engine.TimeFNO1d, factory)
        self.assertIs(engine.penalties, penalty)

    def test_dense_sparse_three_arms_cpu_cuda_exact_resume_and_observed_scaling(self):
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(83)
        x = rng.normal(size=(8, 64)).astype('float32')
        all_y = np.stack([x, .95 * x, .85 * x, .7 * x, .5 * x], 1)
        all_times = [0., .05, .125, .25, .5]
        base = dict(config(), seed=0, steps=4, width=4, modes=4, layers=2,
                    lr=.001, lr_min=.00001, weight_decay=.0001, batch_size=4,
                    eval_every=2, checkpoint_every=2)
        for device in self.devices():
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                initializations = []
                warm_paths = {}
                for condition, observed, evaluation in (
                        ('dense', [0, 1, 2, 3, 4], [1, 2, 3, 4]),
                        ('sparse', [0, 4], [1, 2, 3])):
                    arrays = dict(train_x=x, train_y=all_y[:, observed], val_x=x[:2],
                                  val_y=all_y[:2], times=[all_times[i] for i in observed],
                                  val_times=all_times, grid256=np.arange(64) / 32)
                    cbase = dict(base, evaluation_columns=evaluation)
                    folder = root / condition
                    context = dict(qa='synthetic-only', condition=condition)
                    warm = train_one(dict(cbase, kind='warm', strength=0.), arrays, context,
                                     folder / 'warm', torch.device(device))
                    initializations.append(warm['initial'])
                    warm_path = folder / 'warm/checkpoints/latest.pt'
                    warm_paths[condition] = warm_path
                    warm_cp = torch.load(warm_path, weights_only=False, map_location='cpu')
                    self.assertEqual(warm_cp['scale'], float(np.asarray(arrays['train_y'], dtype=float).std()))
                    self.assertEqual(warm_cp['normalization'],
                                     dict(mean=float(np.asarray(x, dtype=float).mean()),
                                          std=float(np.asarray(x, dtype=float).std())))
                    self.assertIsNone(warm_cp['calibration'])
                    self.assertEqual(len(warm['metrics']['train']['by_time']), len(observed))
                    self.assertEqual(len(warm['metrics']['val']['by_time']), 5)
                    self.assertAlmostEqual(warm['metrics']['train']['primary'],
                                           float(np.mean(warm['metrics']['train']['by_time'][1:])), places=14)
                    self.assertAlmostEqual(warm['metrics']['val']['primary'],
                                           float(np.mean(np.asarray(warm['metrics']['val']['by_time'])[evaluation])),
                                           places=14)
                    if condition == 'sparse':
                        # A dense-supervision warm must not silently supply the
                        # wrong observed-target scale to a sparse continuation.
                        with self.assertRaisesRegex(ValueError, 'observation normalization mismatch'):
                            train_one(dict(cbase, kind='none', strength=0., lr=.0003), arrays,
                                      context, folder / 'wrong_dense_warm', torch.device(device),
                                      warm_path=warm_paths['dense'])
                        self.assertIs(engine.TimeFNO1d, factory)
                        self.assertIs(engine.penalties, penalty)
                    paired = []
                    for kind in ('none', 'full', 'interval_uniform'):
                        c = dict(cbase, kind=kind, strength=0. if kind == 'none' else .3, lr=.0003)
                        first, second = folder / (kind + '_full'), folder / (kind + '_resumed')
                        complete = train_one(c, arrays, context, first, torch.device(device), warm_path=warm_path)
                        self.assertIsNone(train_one(c, arrays, context, second, torch.device(device),
                                                    warm_path=warm_path, stop_after=2))
                        resumed = train_one(c, arrays, context, second, torch.device(device), warm_path=warm_path)
                        for key in ('history', 'initial', 'batch_hash', 'collocation_hash',
                                    'calibration', 'penalty_weight'):
                            self.assert_tree_equal(complete[key], resumed[key])
                        ca = torch.load(first / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                        cb = torch.load(second / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
                        for key in ca:
                            if key != 'seconds':
                                self.assert_tree_equal(ca[key], cb[key])
                        self.assertEqual(ca['step'], 4)
                        self.assertTrue(all(int(v['step']) == 8 for v in ca['optimizer']['state'].values()))
                        self.assertEqual(set(complete['calibration']['penalty_gradient_norms']),
                                         {'full', 'interval_uniform'})
                        self.assertNotIn('reliability_map', ca['identity']['config'])
                        if kind != 'none':
                            self.assertAlmostEqual(complete['history'][0]['effective_gradient_ratio'], .3,
                                                   places=6)
                        paired.append((complete['initial'], complete['batch_hash'],
                                       complete['collocation_hash'], complete['calibration']))
                        self.assertIs(engine.TimeFNO1d, factory)
                        self.assertIs(engine.penalties, penalty)
                    self.assertTrue(all(item == paired[0] for item in paired))
                self.assertEqual(initializations[0], initializations[1])

    def test_sparse_omitted_poison_is_end_to_end_training_calibration_invariant(self):
        from scripts.allen_interval_replication_data import observations
        from scripts.objective_modal_core import sample_weights
        import scripts.physics_stress_core as engine
        factory, penalty = engine.TimeFNO1d, engine.penalties
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(912)
        x = rng.normal(size=(8, 64))
        y = np.stack([x, .95 * x, .85 * x, .7 * x, .5 * x], 1)
        clean = dict(train_x=x, train_y=y, val_x=x[:2], val_y=y[:2],
                     times=[0., .05, .125, .25, .5], grid256=np.arange(64) / 32,
                     train_ids=list(range(200000, 200008)), val_ids=[200100, 200101])
        base = dict(config(), seed=0, steps=2, width=4, modes=4, layers=2,
                    lr=.001, lr_min=.00001, weight_decay=.0001, batch_size=4,
                    eval_every=2, checkpoint_every=2, evaluation_columns=[1, 2, 3])
        original_arrays, original_obs = observations(clean, 'sparse')
        original_weights = sample_weights(original_arrays['train_y'].reshape(-1, 64), 'normalized')
        for device in self.devices():
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reference_warm = None
                reference_continuations = {}
                for variant, poison in [('normal', None), ('extreme_finite', 1e150), ('nan', np.nan)]:
                    changed = dict(clean, train_y=clean['train_y'].copy())
                    if poison is not None:
                        changed['train_y'][:, 1:4] = poison
                    # Validation remains untouched; only forbidden training
                    # labels differ, including deliberately nonfinite values.
                    arrays, obs = observations(changed, 'sparse')
                    self.assert_tree_equal(arrays, original_arrays)
                    self.assert_tree_equal(obs, original_obs)
                    self.assert_tree_equal(sample_weights(arrays['train_y'].reshape(-1, 64), 'normalized'),
                                           original_weights)
                    context = dict(qa='synthetic_sparse_omitted_poison', observations=obs)
                    folder = root / variant
                    train_one(dict(base, kind='warm', strength=0.), arrays, context,
                              folder / 'warm', torch.device(device))
                    warm_path = folder / 'warm/checkpoints/latest.pt'
                    warm = torch.load(warm_path, weights_only=False, map_location='cpu')
                    if reference_warm is None:
                        reference_warm = warm
                    else:
                        for key in warm:
                            if key != 'seconds':
                                self.assert_tree_equal(warm[key], reference_warm[key])
                    for kind in ('none', 'full', 'interval_uniform'):
                        c = dict(base, kind=kind, strength=0. if kind == 'none' else .3, lr=.0003)
                        result = train_one(c, arrays, context, folder / kind,
                                           torch.device(device), warm_path=warm_path)
                        checkpoint = torch.load(folder / kind / 'checkpoints/latest.pt',
                                                weights_only=False, map_location='cpu')
                        if variant == 'normal':
                            reference_continuations[kind] = (result, checkpoint)
                        else:
                            old_result, old_checkpoint = reference_continuations[kind]
                            for key in ('history', 'initial', 'batch_hash', 'collocation_hash',
                                        'calibration', 'penalty_weight'):
                                self.assert_tree_equal(result[key], old_result[key])
                            # Separate warm artifacts have different timing/hash
                            # receipts; all fitted state and random states must
                            # nevertheless be exactly invariant to the poison.
                            for key in ('normalization', 'scale', 'calibration', 'model', 'optimizer',
                                        'rng', 'collocation_rng', 'history', 'initial',
                                        'batch_hash', 'collocation_hash', 'step'):
                                self.assert_tree_equal(checkpoint[key], old_checkpoint[key])
                        self.assertIs(engine.TimeFNO1d, factory)
                        self.assertIs(engine.penalties, penalty)


if __name__ == '__main__':
    unittest.main()
