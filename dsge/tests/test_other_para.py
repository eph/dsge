from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import DSGE

from dsge.examples import sw

class TestOtherPara(TestCase):

    def test_simple(self):

        from dsge.examples import sw
        from dsge.translate import smc

        # print(smc(sw))

        #self.assertAlmostEqual(1,0)
