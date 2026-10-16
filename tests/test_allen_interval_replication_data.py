import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import allen_interval_replication_data as data
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save


class AllenIntervalReplicationDataTests(unittest.TestCase):
    def test_fixed_split_consumed_pool_no_overlap_or_reserves(self):
        split = data.split_ids()
        self.assertEqual(split, data.split_ids(2026101601))
        self.assertNotEqual(split, data.split_ids(2026101602))
        self.assertEqual(len(split['train_ids']), 100)
        self.assertEqual(len(split['val_ids']), 200)
        self.assertEqual(split['train_ids'], sorted(split['train_ids']))
        self.assertEqual(split['val_ids'], sorted(split['val_ids']))
        self.assertFalse(set(split['train_ids']) & set(split['val_ids']))
        self.assertEqual(set(split['train_ids']) | set(split['val_ids']), set(range(200000, 200300)))
        self.assertTrue(set(range(200000, 200100)) & set(split['val_ids']))
        invalid = copy.deepcopy(split)
        invalid['train_ids'][0] = 200300
        with self.assertRaisesRegex(ValueError, 'reserves forbidden'):
            data._validate_split(invalid)
        invalid = copy.deepcopy(split)
        invalid['train_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'requires100'):
            data._validate_split(invalid)

    def make_cache(self, folder):
        root = Path(folder)
        old = dict(tasks=dict(allen_cahn=dict(train_ids=list(range(200000, 200100)),
                    val_ids=list(range(200100, 200300)), times=data.TIMES)))
        atomic_json(root/data.PARENT_PLAN, old)
        for split, key in [('train', 'train_ids'), ('val', 'val_ids')]:
            ids = old['tasks']['allen_cahn'][key]
            for start in range(0, len(ids), 50):
                part = ids[start:start+50]
                x = (np.array(part)[:, None]-200000)/300+np.arange(256)[None, :]/256
                y = x[:, None, :]+np.array(data.TIMES)[None, :, None]*(1+x[:, None, :])
                atomic_save(dict(plan=data._digest(old), ids=part, x=x, y=y),
                            root/data.CACHE_ROOT/f'allen_{split}_{start:04d}.pt')
        return root, old

    def test_read_only_cache_reindexing_digest_and_missing_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root, old = self.make_cache(folder)
            original = {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
            split = data.split_ids()
            # Ordering is explicitly supplied by the plan, not inferred from
            # former train/dev membership or the native cache row order.
            split['train_ids'].reverse()
            with patch.object(data, 'ROOT', root):
                clean, context = data.load_cached_split(split)
                self.assertEqual(len(context['chunks']), 6)
                self.assertFalse(context['new_solver_performed'])
                np.testing.assert_array_equal(clean['train_x'][:, 0],
                                              (np.array(split['train_ids'])-200000)/300)
                self.assertEqual(clean['train_ids'], split['train_ids'])
                self.assertEqual(clean['val_y'].shape, (200, 5, 256))
                after = {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
                self.assertEqual(original, after)
                self.assertEqual(context['parent_plan'], data._digest(old))
                bad = dict(split, parent_plan_sha256='0'*64)
                with self.assertRaisesRegex(ValueError, 'parent plan SHA'):
                    data.load_cached_split(bad)
                path = root/data.CACHE_ROOT/'allen_train_0000.pt'
                import torch
                corrupted = torch.load(path, weights_only=False)
                corrupted['ids'] = corrupted['ids'][::-1]
                atomic_save(corrupted, path)
                with self.assertRaisesRegex(ValueError, 'chunk IDs changed'):
                    data.load_cached_split(split)
                path.unlink()
                before_missing = {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
                with self.assertRaises(FileNotFoundError):
                    data.load_cached_split(split)
                after_missing = {p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*') if p.is_file()}
                self.assertEqual(before_missing, after_missing)

    def test_sparse_poison_omitted_and_development_labels_no_training_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.make_cache(folder)
            with patch.object(data, 'ROOT', root):
                clean, _ = data.load_cached_split(data.split_ids())
            baseline, obs = data.observations(clean, 'sparse')
            self.assertEqual(baseline['times'], [0., .5])
            self.assertEqual(baseline['train_y'].shape, (100, 2, 256))
            self.assertEqual(baseline['val_y'].shape, (200, 5, 256))
            self.assertFalse(obs['heldout_training_labels_accessed_for_stats'])
            self.assertFalse(obs['development_labels_accessed_for_training_stats'])
            changed = copy.deepcopy(clean)
            changed['train_y'][:, 1:4] = np.nan
            poisoned, proof = data.observations(changed, 'sparse')
            self.assertEqual(obs, proof)
            for name in ['train_x', 'train_y', 'times', 'grid256']:
                np.testing.assert_array_equal(baseline[name], poisoned[name])
            changed = copy.deepcopy(clean)
            changed['val_y'][:] = np.nan
            poisoned, proof = data.observations(changed, 'sparse')
            self.assertEqual(obs, proof)
            np.testing.assert_array_equal(baseline['train_x'], poisoned['train_x'])
            np.testing.assert_array_equal(baseline['train_y'], poisoned['train_y'])
            self.assertTrue(np.isnan(poisoned['val_y']).all())  # evaluation values are not silently replaced
            dense, dense_obs = data.observations(clean, 'dense')
            changed = copy.deepcopy(clean)
            changed['train_y'][:, 1:4] += 100
            dense_changed, changed_obs = data.observations(changed, 'dense')
            self.assertNotEqual(dense_obs['selected_target_scale'], changed_obs['selected_target_scale'])
            self.assertNotEqual(dense_obs['normalization_sha256'], changed_obs['normalization_sha256'])
            self.assertFalse(np.array_equal(dense['train_y'], dense_changed['train_y']))
            self.assertEqual(dense_obs['input_normalization'], changed_obs['input_normalization'])
            self.assertEqual(json.loads(json.dumps(obs)), obs)


if __name__ == '__main__':
    unittest.main()
