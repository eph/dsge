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
    c_para[1] = e_para[16]   # dpstar (no clear mapping given, using provided constant)
    c_para[2] = e_para[15]          # yg (no clear mapping given, using provided constant)
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
    c_para[22] = e_para[21]
    c_para[23] = e_para[17]
    c_para[24] = e_para[18]
    c_para[25] = e_para[19]
    c_para[26] = e_para[20]

    return c_para






class TestFHP(TestCase):

    def setUp(self):

        model_file = pkg_resources.resource_filename('dsge', 'examples/fhp/fhp.yaml')
        self.model = FHPRepAgent.read(model_file)

        pe_model_file = pkg_resources.resource_filename('dsge', 'examples/fhp/partial_equilibrium.yaml')
        self.pe_model = FHPRepAgent.read(pe_model_file)

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

    def test_expectations(self):
        compiled_model = self.model.compile_model(k=1000,expectations=1)
        p0 = self.model.p0()
        irfs = compiled_model.impulse_response(p0, 10)

        assert_array_almost_equal(irfs['e_mu']['pi(1)'].shift(1).dropna(),
                                  irfs['e_mu']['pi'].iloc[1:])

        compiled_model = self.model.compile_model(k=1000,expectations=3)
        p0 = self.model.p0()
        irfs = compiled_model.impulse_response(p0, 10)

        assert_array_almost_equal(irfs['e_mu']['pi(3)'].shift(3).dropna(),
                                  irfs['e_mu']['pi'].iloc[3:])

    def test_expectations_2(self):
        compiled_model = self.pe_model.compile_model(k=1, expectations=1)
        p0 = self.pe_model.p0()
        irf = compiled_model.impulse_response(p0,10)['e_y']
        cal = dict(zip(compiled_model.parameter_names, p0))
        assert_array_almost_equal(cal['rho']*cal['kappa']*irf.y + cal['beta']*irf.vp,
                                  irf['pi(1)'])

        k = 4
        compiled_model = self.pe_model.compile_model(k=k, expectations=1)
        p0 = self.pe_model.p0()
        irf = compiled_model.impulse_response(p0,10)['e_y']
        cal = dict(zip(compiled_model.parameter_names, p0))
        Ak = sum([(cal['beta']*cal['rho'])**ki for ki in range(k)])
        assert_array_almost_equal(cal['kappa']*cal['rho']*Ak*irf.y + cal['beta']**k*irf.vp,
                                  irf['pi(1)'])

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

    def test_lik(self):
        p0 = self.model.p0()
        print(p0)
        compiled_model = self.model.compile_model(k=4)
        print(compiled_model.log_lik(p0))

    def test_compile(self):
        from dsge.FHPRepAgent import make_fortran_model
        make_fortran_model(self.model)
    def test_fortran(self):

        p0 = self.model.p0()
        compiled_model = self.model.compile_model(k=4)

        alpha0_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alpha0_cycle.txt')
        alpha0_cycle_py = compiled_model.alpha0_cycle(p0)
        assert_array_almost_equal(alpha0_cycle, alpha0_cycle_py)

        # do same for alpha1_cycle, beta0_cycle, alphaC_cycle, alphaF_cycle, betaS_cycle
        alpha1_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alpha1_cycle.txt')
        alpha1_cycle_py = compiled_model.alpha1_cycle(p0)
        assert_array_almost_equal(alpha1_cycle, alpha1_cycle_py)

        beta0_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/beta0_cycle.txt')
        beta0_cycle_py = compiled_model.beta0_cycle(p0)
        assert_array_almost_equal(beta0_cycle, beta0_cycle_py)

        alphaC_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alphaC_cycle.txt')
        alphaC_cycle_py = compiled_model.alphaC_cycle(p0)
        assert_array_almost_equal(alphaC_cycle, alphaC_cycle_py)

        alphaF_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alphaF_cycle.txt')
        alphaF_cycle_py = compiled_model.alphaF_cycle(p0)
        assert_array_almost_equal(alphaF_cycle, alphaF_cycle_py)

        betaS_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/betaS_cycle.txt')
        betaS_cycle_py = compiled_model.betaS_cycle(p0)
        assert_array_almost_equal(betaS_cycle, betaS_cycle_py)

        alpha0_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alpha0_trend.txt')
        alpha0_trend_py = compiled_model.alpha0_trend(p0)
        assert_array_almost_equal(alpha0_trend, alpha0_trend_py)

        alpha1_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alpha1_trend.txt')
        alpha1_trend_py = compiled_model.alpha1_trend(p0)
        assert_array_almost_equal(alpha1_trend, alpha1_trend_py)

        betaV_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/betaV_trend.txt')
        betaV_trend_py = compiled_model.betaV_trend(p0)
        assert_array_almost_equal(betaV_trend, betaV_trend_py)

        alphaC_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alphaC_trend.txt')
        alphaC_trend_py = compiled_model.alphaC_trend(p0)
        assert_array_almost_equal(alphaC_trend, alphaC_trend_py)

        alphaF_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alphaF_trend.txt')
        alphaF_trend_py = compiled_model.alphaF_trend(p0)
        assert_array_almost_equal(alphaF_trend, alphaF_trend_py)

        alphaB_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/alphaB_trend.txt')
        alphaB_trend_py = compiled_model.alphaB_trend(p0)
        assert_array_almost_equal(alphaB_trend, alphaB_trend_py)

        value_gammaC = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/value_gammaC.txt')
        value_gammaC_py = compiled_model.value_gammaC(p0)
        assert_array_almost_equal(value_gammaC, value_gammaC_py)

        value_gamma = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/value_gamma.txt')
        value_gamma_py = compiled_model.value_gamma(p0)
        assert_array_almost_equal(value_gamma, value_gamma_py)

        value_Cx = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/value_Cx.txt')
        value_Cx_py = compiled_model.value_Cx(p0)
        assert_array_almost_equal(value_Cx, value_Cx_py)

        value_Cs = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/value_Cs.txt')
        value_Cs_py = compiled_model.value_Cs(p0)
        assert_array_almost_equal(value_Cs, value_Cs_py, 6)

        compiled_model.log_lik(p0)
        A_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/A_cycle.txt')
        A_cycle_py = compiled_model.A_cycle
        assert_array_almost_equal(A_cycle, A_cycle_py, 6)

        B_cycle = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/B_cycle.txt')
        B_cycle_py = compiled_model.B_cycle
        assert_array_almost_equal(B_cycle, B_cycle_py, 6)

        A_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/A_trend.txt')
        A_trend_py = compiled_model.A_trend
        assert_array_almost_equal(A_trend, A_trend_py, 6)

        B_trend = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/B_trend.txt')
        B_trend_py = compiled_model.B_trend
        assert_array_almost_equal(B_trend, B_trend_py, 6)

        CC_py, TT_py, RR_py, QQ_py, DD_py, ZZ_py, HH_py = compiled_model.system_matrices(p0)
        TT = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/TT.txt')
        assert_array_almost_equal(TT, TT_py, 6)

        RR = np.loadtxt('/home/eherbst/Dropbox/code/dsge/dsge/tests/fhp_fortran_matrices/RR.txt')
        assert_array_almost_equal(RR, RR_py, 6)
