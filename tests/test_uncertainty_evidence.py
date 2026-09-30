import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
from scripts.run_uncertainty_modal import compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/public_uncertainty_modal_v1'


class UncertaintyEvidenceTest(unittest.TestCase):
    def test_seal_statistics_and_paper(self):
        lock=json.loads((OUT/'selection_lock.json').read_text())
        self.assertEqual(lock,json.loads(subprocess.check_output(['git','show','bfd52cd:results/public_uncertainty_modal_v1/selection_lock.json'],cwd=ROOT)))
        paper=(ROOT/'paper/FOLLOWUP_UNCERTAINTY_20260930.md').read_text(encoding='utf-8')
        for task in lock['tasks']:
            s=json.loads((OUT/task/'development.json').read_text());c=json.loads((OUT/task/'confirmation.json').read_text())
            self.assertTrue(c['git_commit'].startswith('bfd52cd'))
            self.assertEqual(c['selected'],s['selected'])
            for arm in ('all','old_shrink','uncertainty','matched_global'):
                self.assertIn(f"{c['means'][arm]:.{12 if task=='advection' else 6}f}",paper)
            for arm in ('none','old_shrink','all','matched_global'):
                self.assertEqual(compare(c['errors'][c['selected']],c['errors'][arm]),c['comparisons'][arm])
                if arm!='none':self.assertFalse(c['comparisons'][arm]['success'])
            self.assertEqual(c['comparisons']['none']['success'],task in ('burgers_independent','allen_cahn'))
            for arm,values in c['errors'].items():self.assertEqual(float(np.mean(values)),c['means'][arm])


if __name__=='__main__':unittest.main()
