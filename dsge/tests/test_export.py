import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.translate import translate
try:
    from importlib.resources import files, as_file
except Exception:  # pragma: no cover
    from importlib_resources import files, as_file  # type: ignore

class TestTranslate(TestCase):



    def test_simple(self):
        with as_file(files('dsge') / 'examples' / 'sw' / 'sw.yaml') as p:
            sw = read_yaml(str(p))
        #translate(sw, output_dir='/mq/home/m1eph00/tmp/sw_test/')
        pass
