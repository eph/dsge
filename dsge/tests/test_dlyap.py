import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

#from dsge.dlyap import dlyap as dlyap2

#from dsge.fortran import dlyap
from dsge import DSGE


class TestDLYAP(TestCase):

    def test_simple(self):
        RR = np.eye(3)
        TT = np.zeros((3,3))

        #PP, RC = dlyap.dlyap(TT,RR)
        #self.assertEqual(RC,0)
        #assert_equal(np.eye(3),PP)


    # def test_sw(self):
    #     sw = DSGE.DSGE.read('dsge/examples/sw/sw.yaml')
    #     sw = sw.compile_model()

    #     p0 = [0.1657,0.7869,0.5509,0.1901,1.3333,1.6064,5.7606,0.72,0.7,1.9,0.65,0.57,0.3,0.5462,2.0443,0.8103,0.0882,0.2247,0.9577,0.2194,0.9767,0.7113,0.1479,0.8895,0.9688,0.5,0.72,0.85,0.4582,0.24,0.5291,0.4526,0.2449,0.141,0.2446]

    #     TT, RR, QQ, DD, ZZ, HH = sw.system_matrices(p0)

    #     P0, info = dlyap.dlyap(TT,RR.dot(QQ).dot(RR.T))

    #     P02, info = dlyap2(TT,RR.dot(QQ).dot(RR.T))

    #     assert_array_almost_equal(P0,P02)
