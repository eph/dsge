import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.fortran import gensysw

class TestGensys(TestCase):

    def test_simple(self):
        RR = np.eye(3)
        TT = np.zeros((3,3))


        TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys_wrapper.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)

        #self.assertEqual(RC,0)
        #assert_equal(np.eye(3),PP)
