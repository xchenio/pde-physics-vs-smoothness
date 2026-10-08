import unittest
import numpy as np
from scripts.confirm_fno_regularization import contrast


class ConfirmationTests(unittest.TestCase):
    def test_effect_gate_pairing_and_reproducibility(self):
        b=np.linspace(.1,.9,50)
        win=contrast(b*.98,b,replicates=1000)
        self.assertTrue(win['success']);self.assertAlmostEqual(win['relative_gain'],.02)
        self.assertEqual(win,contrast(b*.98,b,replicates=1000))
        self.assertFalse(contrast(b*.999,b,replicates=1000)['success'])
        self.assertFalse(contrast(b*1.02,b,replicates=1000)['success'])
        with self.assertRaises(ValueError):contrast(b,b[:20])


if __name__=='__main__':unittest.main()
