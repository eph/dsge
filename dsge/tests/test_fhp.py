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


def e2c_para(e_para, k):

    # Initialize Chris parameter vector with zeros
    c_para = np.zeros((27))

    # Mapping ED parameters to Chris parameters based on variable names
    c_para[0] = e_para[8]    # ra <- r_A
    c_para[1] = 2.0          # dpstar (no clear mapping given, using provided constant)
    c_para[2] = 0.45          # yg (no clear mapping given, using provided constant)
    c_para[3] = k            # kk <- k
    c_para[4] = e_para[0]    # sigma <- sigma
    c_para[5] = e_para[4]    # alpha <- alpha
    c_para[6] = e_para[3]    # delta <- delta
    c_para[7] = e_para[1]

    c_para[8] = e_para[7]    # kappa <- kappa
    c_para[9] = e_para[2]    # phik <- phi_k
    c_para[10] = e_para[5]   # phidp <- phi_pi
    c_para[11] = e_para[6]   # phiy <- phi_y
    c_para[12] = e_para[5]   # phidp <- phi_pi
    c_para[13] = e_para[6]   # phiy <- phi_y


    # phidpbar, phiybar (no clear mapping given, leaving as 0)
    c_para[14] = e_para[10]  # gammah, gammadp, gammak is assumed to be gamma
    c_para[15] = e_para[10]
    c_para[16] = e_para[10]
    c_para[17] = e_para[11]  # rhor <- rho_re
    c_para[18] = e_para[9]   # rhomu <- rho_mu
    c_para[19] = e_para[12]  # rhochi <- rho_chi
    c_para[20] = e_para[13]  # rhog <- rho_g
    c_para[21] = e_para[14]  # rhom <- rho_mp
    c_para[22:] = 0.01

    return c_para






class TestFHP(TestCase):

    def setUp(self):

        model_file = pkg_resources.resource_filename('dsge', 'examples/fhp/fhp.yaml')
        self.model = FHPRepAgent.read(model_file)

    def test_load(self):
        pass

    def test_compile(self):
        self.compiled_model = self.model.compile_model()


    def test_irf_re(self):
        p0 = self.model.p0()
        compiled_model = self.model.compile_model(k=12000)
        TT = 10
        ed_irf = compiled_model.impulse_response(p0, TT)
        from . import test_fhp_re
        c_para = e2c_para(p0, 12000)
        chris_irf = test_fhp_re.rational_expectations_irf(c_para,TT)

        # assert equalities of two irfs

        keys = ['e_re', 'e_mu', 'e_chi', 'e_g', 'e_mp']
        values = ['epr','epmu','epchi','epg','epm']
        e2c_shocks = dict(zip(keys, values))

        e2c_vars = {'y': 'yy',
                    'r': 'nr',
                    'i': 'inv',
                    'kp': 'kp',
                    'c': 'cc',
                    'pi': 'dp',
                    'mc': 'mc'}

        e_vars = e2c_vars.keys()
        c_vars = e2c_vars.values()
        for e_shock, c_shock in e2c_shocks.items():
            # check if equal
            assert_array_almost_equal(ed_irf[e_shock][e_vars],
                                      chris_irf[c_shock][c_vars], decimal=6)

    def test_irf_fhp(self):
        k, TT = 0, 10
        p0 = self.model.p0()
        compiled_model = self.model.compile_model(k=k)
        ed_irf = compiled_model.impulse_response(p0, TT)

        from . import test_fhp_re
        c_para = e2c_para(p0, k)
        chris_irf = test_fhp_re.fhp_irf(c_para,TT+1)

        keys = ['e_re', 'e_mu', 'e_chi', 'e_g', 'e_mp']
        values = ['epr','epmu','epchi','epg','epm']
        e2c_shocks = dict(zip(keys, values))

        e2c_vars = {'y': 'yy',
                    'r': 'nr',
                    'i': 'inv',
                    'kp': 'kp',
                    'c': 'cc',
                    'pi': 'dp',
                    'mc': 'mc'}

        e2c_vars = {'y_cycle': 'yytil',
                    'r_cycle': 'nrtil',
                    'i_cycle': 'invtil',
                    'kp_cycle': 'kptil',
                    'c_cycle': 'cctil',
                    'pi_cycle': 'dptil',
                    'mc_cycle': 'mctil',
                    'y': 'yy',
                    'r': 'nr',
                    'i': 'inv',
                    'kp': 'kp',
                    'c': 'cc',
                    'pi': 'dp',
                    'mc': 'mc'}


        e_vars = e2c_vars.keys()
        c_vars = e2c_vars.values()
        for e_shock, c_shock in e2c_shocks.items():

            # check if equal
            #print(ed_irf[e_shock][e_vars].values-chris_irf[c_shock][c_vars].values)
            assert_array_almost_equal(ed_irf[e_shock][e_vars],
                                      chris_irf[c_shock][c_vars], decimal=6)
