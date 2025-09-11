import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.translate import translate
from dsge.resource_utils import resource_path

class TestTranslate(TestCase):



    def test_simple(self):
        with resource_path('examples/sw/sw.yaml') as p:
            sw = read_yaml(str(p))
        #translate(sw, output_dir='/mq/home/m1eph00/tmp/sw_test/')
        pass
