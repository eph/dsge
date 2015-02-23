import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.fortran import dlyap

class TestDLYAP(TestCase):

    def test_simple(self):
        RR = np.eye(3)
        TT = np.zeros((3,3))

        PP, RC = dlyap.dlyap(TT,RR)
        self.assertEqual(RC,0)
        assert_equal(np.eye(3),PP)
