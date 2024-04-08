import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.gensys import gensys
from numpy.testing import assert_equal

class TestLinver(TestCase):

    def test_linver_gensys(self):
        G0 = np.loadtxt('dsge/tests/linver/G0.txt')
        G1 = np.loadtxt('dsge/tests/linver/G1.txt')
        PSI = np.loadtxt('dsge/tests/linver/PSI.txt')
        PI = np.loadtxt('dsge/tests/linver/PI.txt')

        TT, RR, RC = gensys(G0, G1, PSI, PI)

        
