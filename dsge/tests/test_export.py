import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.DSGE import DSGE
from dsge.translate import translate

class TestTranslate(TestCase):



    def test_simple(self):
        sw = DSGE.read('dsge/examples/sw/sw.yaml')
        #translate(sw, output_dir='/mq/home/m1eph00/tmp/sw_test/')
        pass
