import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import allen_interval_reference as reference
from scripts.allen_cahn_data import reaction
from scripts.allen_cahn_trajectory_v1 import trajectories


class AllenIntervalReferenceTests(unittest.TestCase):
    def test_short_constant_solution_units_continuous_shrinkage(self):
        centers, hs = [.001], [.0005]
        times = reference.interval_times(centers, hs)
        arrays = [trajectories(np.full((2, n), .25), times, dt=.000125)
                  for n in [128, 128, 64]]
        result = reference.compute_map(*arrays, times, .4, centers=centers,
                                       half_widths=hs, final_time=.5, actual_n=32, band=4)
        self.assertEqual(result['weights64'].shape, (1, 1, 5))
        self.assertEqual(result['tau'], .0008)
        self.assertEqual(result['weights64'].dtype, np.float64)
        np.testing.assert_array_equal(result['weights32'], result['weights64'].astype(np.float32))
        eps = np.maximum.reduce(list(result['diagnostics']['rms_rate_components'].values()))
        np.testing.assert_array_equal(result['epsilon'], eps)
        np.testing.assert_array_equal(result['weights64'], result['tau']**2/(result['tau']**2+eps**2))
        self.assertTrue(np.all(result['weights64'] > 0))
        self.assertTrue(np.all(result['weights64'] <= 1))
        self.assertLess(result['epsilon'].max(), 1e-8)
        for key, a in [('weight64_statistics', result['weights64']),
                       ('weight32_statistics', result['weights32'])]:
            stats = result['diagnostics'][key]
            self.assertEqual(stats['all_ones_exact'], bool(np.all(a == 1)))
            self.assertEqual(stats['unique_count'], np.unique(a).size)
            self.assertAlmostEqual(stats['mean'], float(a.astype(float).mean()))
            self.assertAlmostEqual(stats['parseval_weighted_mass_fraction'],
                                   float(np.sum(a*np.array([1, 2, 2, 2, 2]))/9))
        nodes = np.linspace(.0005, .0015, 9)
        expected_rate = (reaction(.25, nodes[-1])-reaction(.25, nodes[0]))/.001
        self.assertGreater(expected_rate, 1.)

    def test_explicit_per_mode_rms_max_not_percentage_or_median(self):
        centers, hs = [.01], [.005]
        times = reference.interval_times(centers, hs)
        x = np.arange(128)/128
        temporal = np.sin(2*np.pi*x)[None, None, :]*times[None, :, None]
        fine = np.repeat(temporal, 2, axis=0)
        coarse = fine.copy()
        coarse[1] *= 2
        spatial = fine[..., ::2]
        result = reference.compute_map(fine, coarse, spatial, times, .4, centers=centers,
                                       half_widths=hs, actual_n=32, band=4)
        term = result['diagnostics']['rms_rate_components']['timestep_endpoint_rate']
        self.assertAlmostEqual(term[0, 0, 1], np.sqrt(.5**2/2), places=12)
        self.assertFalse(result['diagnostics']['median_rescaling'])
        self.assertFalse(result['diagnostics']['hard_thresholding'])
        self.assertLess(result['weights64'][0, 0, 1], .001)
        self.assertGreater(result['weights64'][0, 0, 0], .99)
        with self.assertRaisesRegex(ValueError, 'float32.*underflow'):
            reference.compute_map(fine, coarse, spatial, times, 1e-25, centers=centers,
                                  half_widths=hs, actual_n=32, band=4)

    def test_analytic_reaction_and_nonzero_reference_errors_at_small_scale(self):
        centers, hs = [.2], [.1]
        times = reference.interval_times(centers, hs)
        # Exact spatially constant nonlinear PDE solution. The two comparison
        # references have known small numerical-error surrogates starting at0.
        fine = np.broadcast_to(reaction(.25, times)[None, :, None], (2, len(times), 64)).copy()
        coarse = fine+1e-5*times[None, :, None]**2
        spatial = fine[..., ::2]+2e-5*times[None, :, None]**3
        result = reference.compute_map(fine, coarse, spatial, times, 1e-3,
                                       centers=centers, half_widths=hs, actual_n=16, band=4)
        terms = result['diagnostics']['rms_rate_components']
        self.assertGreater(terms['timestep_endpoint_rate'][0, 0, 0], 0)
        self.assertGreater(terms['timestep_rhs_average'][0, 0, 0], 0)
        self.assertGreater(terms['space_endpoint_rate'][0, 0, 0], 0)
        self.assertGreater(terms['space_rhs_average'][0, 0, 0], 0)
        self.assertGreater(terms['simpson5_vs9_rhs_average'][0, 0, 0], 0)
        self.assertAlmostEqual(terms['timestep_endpoint_rate'][0, 0, 0], 4e-6, places=13)
        self.assertEqual(result['tau'], 2e-6)
        expected = result['tau']**2/(result['tau']**2+result['epsilon']**2)
        np.testing.assert_array_equal(result['weights64'], expected)
        self.assertLess(result['weights64'][0, 0, 0], .5)
        self.assertFalse(result['diagnostics']['median_rescaling'])
        self.assertIn('cancelled', result['diagnostics']['global_scale_note'])

    def test_train_only_cache_resume_hashes_and_identity(self):
        def constant_initial(ids, n):
            return np.full((len(ids), n), .25)

        calls = []

        def interrupted_solver(x, times, **kwargs):
            calls.append(kwargs['dt'])
            if len(calls) == 2:
                raise RuntimeError('simulated interruption')
            return np.broadcast_to(reaction(x[:, None], times[None, :, None]),
                                   (len(x), len(times), x.shape[-1])).copy()

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(reference, 'initial', side_effect=constant_initial):
                with patch.object(reference, 'trajectories', side_effect=interrupted_solver):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        reference.build_reference(folder, reference.TRAIN_IDS, .4, {'qa': 'mock-only'})
                self.assertTrue((Path(folder)/'native_fine_chunk_0000.pt').exists())
                with patch.object(reference, 'trajectories', side_effect=lambda x, times, **kwargs: np.broadcast_to(reaction(x[:, None], times[None, :, None]), (len(x), len(times), x.shape[-1])).copy()) as resumed:
                    obj = reference.build_reference(folder, reference.TRAIN_IDS, .4, {'qa': 'mock-only'})
                self.assertEqual(resumed.call_count, 5)
            with patch.object(reference, 'trajectories', side_effect=AssertionError('unexpected regeneration')):
                replay = reference.build_reference(folder, reference.TRAIN_IDS, .4, {'qa': 'mock-only'})
            self.assertEqual(replay['payload_sha256'], obj['payload_sha256'])
            self.assertTrue(obj['provenance']['training_only_extra_solver_information'])
            self.assertTrue(obj['provenance']['no_development_or_reserved_access'])
            self.assertEqual(len(obj['hashes']['stages']), 6)
            with self.assertRaisesRegex(ValueError, 'folder identity changed'):
                reference.build_reference(folder, reference.TRAIN_IDS, .5, {'qa': 'mock-only'})
            with self.assertRaisesRegex(ValueError, 'dev/reserves forbidden'):
                reference.build_reference(folder, list(range(200100, 200116)), .4, {})
            changed = copy.deepcopy(obj)
            changed['weights64'][0, 0, 0] *= .9
            with self.assertRaisesRegex(ValueError, 'weight32|shrinkage|payload'):
                reference._validate_map(changed, obj['identity'], Path(folder))
            stage = Path(folder)/next(iter(obj['hashes']['stages']))
            original = stage.read_bytes()
            stage.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'file SHA-256 changed'):
                reference.build_reference(folder, reference.TRAIN_IDS, .4, {'qa': 'mock-only'})


if __name__ == '__main__':
    unittest.main()
