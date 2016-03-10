from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge.fortran import gensysw
from dsge.DSGE import DSGE

import pkg_resources

class TestGensys(TestCase):

    def test_simple(self):
        RR = np.eye(3)
        TT = np.zeros((3,3))


        #TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensysw.gensys_wrapper.call_gensys(G0, G1, C0, PSI, PPI, 1.00000000001)

        #self.assertEqual(RC,0)
        #assert_equal(np.eye(3),PP)

    def test_pc(self):
        relative_loc = ('examples/schorf_phillips_curve/'
                        'schorf_phillips_curve.yaml')
        model_file = pkg_resources.resource_filename('dsge', relative_loc)


        pc = DSGE.read(model_file)
        p0 = pc.p0()
        model = pc.compile_model()

        TT,RR,RC = model.solve_LRE(p0)

        assert_array_almost_equal(TT, np.zeros_like(TT))

        para = dict(zip(map(str,pc.parameters),p0))
        kap, tau, psi = para['kap'], para['tau'], para['psi']

        RRexact = ( (1/(1+kap*tau*psi)) *
                    np.array([[-tau, 1, -tau*psi],
                              [-kap*tau, kap, 1],
                              [1, kap*psi, psi]]))
        assert_array_almost_equal(RR[:3,:3], RRexact)

    # def test_single_equation(self):
    #     simple = DSGE.read('dsge/examples/simple-model/simple_model_est.yaml')
    #     p0 = simple.p0()

    #     model = simple.compile_model()
    #     print model.GAM0(p0)
    #     print model.GAM1(p0)
    #     print model.PSI(p0)
    #     print model.PPI(p0)
    #     print model.solve_LRE(p0)
    #     print fjdksalf
