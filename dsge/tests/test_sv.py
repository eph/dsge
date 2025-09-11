#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge.DSGE import DSGE

from dsge import parse_yaml

# class TestSV(TestCase):

#     # startup
#     def setUp(self):
#         relative_loc = 'examples/dsge_sv.yaml'
#         model_file = pkg_resources.resource_filename('dsge', relative_loc)
#         self.model = parse_yaml(relative_loc)

#     def test1(self):
#         self.model.python_sims_matrices()
#         print(self.model.HH(self.model['par_ordering']))
#         linear_model = self.model.compile_model()
