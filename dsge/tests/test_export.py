import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.DSGE import DSGE
from dsge.translate import translate

import pkg_resources

class TestTranslate(TestCase):



    def test_simple(self):
        relative_loc = 'examples/sw/sw.yaml'
        model_file = pkg_resources.resource_filename('dsge', relative_loc)
        sw = DSGE.read(model_file)
        #translate(sw, output_dir='/mq/home/m1eph00/tmp/sw_test/')
        pass
