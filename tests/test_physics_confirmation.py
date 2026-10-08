import unittest
import numpy as np
from scripts.confirm_physics_token import contrast


class ConfirmationTests(unittest.TestCase):
    def test_registered_effect_and_multiplicity_gate(self):
        b=np.ones(50);self.assertTrue(contrast(b*.9,b,replicates=100)['success'])
        self.assertFalse(contrast(b*.995,b,replicates=100)['success'])
        self.assertFalse(contrast(b,b,replicates=100)['success'])
        self.assertFalse(contrast(b*1.1,b,replicates=100)['success'])
        c=contrast(b*.9,b,replicates=100);np.testing.assert_allclose(c['interval'],[-.1,-.1]);self.assertAlmostEqual(c['relative_gain'],.1)


if __name__=='__main__':unittest.main()
