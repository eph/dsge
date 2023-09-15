#!/usr/bin/env python3
import numpy as np
import pandas as pd

from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge.FHPRepAgent import FHPRepAgent

import pkg_resources



def print_matrix(A, padding=20):
    nr, nc = A.shape
    for i in range(nr):
        for j in range(nc):
            print(f"{str(A[i,j]):<{padding}}", end="")
        print()

class TestFHP(TestCase):

    def setUp(self):

        model_file = pkg_resources.resource_filename('dsge', 'examples/fhp/fhp.yaml')
        self.model = FHPRepAgent.read(model_file)

    def test_load(self):
        pass

    def test_compile(self):
        self.compiled_model = self.model.compile_model()


    def test_irf(self):
        p0 = self.model.p0()
        compiled_model = self.model.compile_model()
        compiled_model.system_matrices(p0)
        irf = compiled_model.impulse_response(p0, 10)

        print(100*irf['e_g'])


        import matplotlib.pyplot as plt


        ax = (100*irf['e_g'][['q','kp','i']]).plot(subplots=True)

        from . import test_fhp_re
        (test_fhp_re.xirf1[['qq','kp','inv']]).plot(ax=ax, subplots=True, linestyle='dashed')
        #(test_fhp_re.xirf_re[['qq','kp','inv']]).plot(ax=ax, subplots=True, linestyle='dotted')
        plt.show()
