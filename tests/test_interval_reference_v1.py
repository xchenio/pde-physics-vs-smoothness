import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from scripts import interval_reference_v1 as reference
from scripts.allen_cahn_data import reaction
from src.public_benchmark import file_hash


class IntervalReferenceTests(unittest.TestCase):
    def plan(self, task, ids):
        times = [0., .0005, .001] if task == 'allen_cahn' else [0., .02, .04]
        return dict(source={'scripts/interval_reference_v1.py': file_hash(Path(reference.__file__))},
                    dev_ids={task: ids},
                    tasks={task: dict(label_times=[times[0], times[-1]],
                                      windows=[dict(reference_times=times, model_times=times)])},
                    task_spec={task: dict(pde=task, length=2., diffusion=.01, reaction=5.)})

    def test_constant_allen_three_reference_resolutions_and_cached_identity(self):
        p = self.plan('allen_cahn', [300000, 300001])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(reference, 'OUTDATA_ROOT', Path(folder)):
                with patch.object(reference, 'initial', side_effect=lambda ids, n: np.full((len(ids), n), .25)):
                    obj, first_hash = reference.build_reference(p, 'allen_cahn')
                expected = np.array([reaction(.25, t) for t in obj['times']])
                for values in [*obj['fields'].values(), *obj['coarse_fields'].values()]:
                    np.testing.assert_allclose(values, np.broadcast_to(expected[None, :, None], values.shape), atol=1e-14)
                    self.assertEqual(values.dtype, np.float64)
                with patch.object(reference, 'trajectories', side_effect=AssertionError('cache unexpectedly recomputed')):
                    replay, replay_hash = reference.build_reference(p, 'allen_cahn')
                self.assertEqual(first_hash, replay_hash)
                self.assertEqual(replay['provenance']['fine_dt'], .0000625)
                self.assertEqual(replay['provenance']['coarse_dt'], .000125)
                changed = copy.deepcopy(p)
                changed['dev_ids']['allen_cahn'].reverse()
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    reference.build_reference(changed, 'allen_cahn')
                changed = copy.deepcopy(p)
                changed['source']['scripts/interval_reference_v1.py'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'sealed reference source changed'):
                    reference.build_reference(changed, 'allen_cahn')

    def test_resume_preserves_completed_stage_inside_eight_case_chunk(self):
        p = self.plan('allen_cahn', list(range(300000, 300010)))
        calls = []

        def fake_solver(x, times, **kwargs):
            calls.append((x.shape, kwargs['dt']))
            if len(calls) == 2:
                raise RuntimeError('simulated interruption')
            return np.repeat(x[:, None, :], len(times), axis=1)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(reference, 'OUTDATA_ROOT', Path(folder)), patch.object(reference, 'trajectories', side_effect=fake_solver):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    reference.build_reference(p, 'allen_cahn')
                chunk = Path(folder) / 'allen_cahn_chunk_0000.pt'
                self.assertTrue(chunk.exists())
                calls.clear()
                with patch.object(reference, 'trajectories', side_effect=lambda x, times, **kwargs: np.repeat(x[:, None, :], len(times), axis=1)) as resumed:
                    obj, _ = reference.build_reference(p, 'allen_cahn')
                self.assertEqual(resumed.call_count, 5)  # six stages minus the saved first stage
                self.assertEqual(obj['fields'][1024].shape, (10, 3, 1024))
                self.assertEqual(len(obj['chunk_sha256']), 2)

    def test_hdf_native_snapshot_selection_time_alignment_and_sha_guard(self):
        p = self.plan('burgers', [2, 0])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'tiny_reference.hdf5'
            values = np.arange(3 * 5 * 1024, dtype=np.float32).reshape(3, 5, 1024)
            with h5py.File(source, 'w') as f:
                f.create_dataset('tensor', data=values)
                f.create_dataset('t-coordinate', data=np.arange(5, dtype=np.float32) * .01)
            p['dataset'] = dict(path=str(source), sha256=file_hash(source))
            with patch.object(reference, 'OUTDATA_ROOT', Path(folder) / 'reference_cache'):
                obj, _ = reference.build_reference(p, 'burgers')
                np.testing.assert_array_equal(obj['fields'][1024], values[[2, 0]][:, [0, 2, 4]].astype(float))
                np.testing.assert_array_equal(obj['fields'][256], obj['fields'][1024][..., ::4])
                np.testing.assert_array_equal(obj['times'], [0., .02, .04])
                self.assertEqual(reference.time_indices(obj['times'], [.02]).tolist(), [1])
                with self.assertRaisesRegex(ValueError, 'not present'):
                    reference.time_indices(obj['times'], [.021])
            wrong = copy.deepcopy(p)
            wrong['dataset']['sha256'] = '0' * 64
            with patch.object(reference, 'OUTDATA_ROOT', Path(folder) / 'wrong_sha'):
                with self.assertRaisesRegex(ValueError, 'dataset SHA-256 mismatch'):
                    reference.build_reference(wrong, 'burgers')
            wrong = copy.deepcopy(p)
            wrong['tasks']['burgers']['windows'][0]['reference_times'].append(.015)
            with self.assertRaisesRegex(ValueError, 'do not align'):
                reference.reference_times(wrong, 'burgers')


if __name__ == '__main__':
    unittest.main()
