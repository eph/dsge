from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from unittest import TestCase

from dsge import DSGE

import pkg_resources

class TestFilter(TestCase):

    def test_sw_lik(self):
        from dsge.examples import sw

        sw = sw.compile_model()

        p0 = [0.1657,0.7869,0.5509,0.4312,0.1901,1.3333,1.6064,5.7606,0.72,0.7,1.9,0.65,0.57,0.3,0.5462,2.0443,0.8103,0.0882,0.2247,0.9577,0.2194,0.9767,0.7113,0.1479,0.8895,0.9688,0.5,0.72,0.85,0.4582,0.24,0.5291,0.4526,0.2449,0.141,0.2446]
        lik0 = sw.log_lik(p0)

        self.assertAlmostEqual(-829.7412615500879, lik0, places=6)

    def test_ar1(self):
        relative_loc = 'examples/ar1/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'ar1.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'arma23_sim200.txt')
        ar1 = DSGE.DSGE.read(model_file)
        ar1['__data__']['estimation']['data'] = data_file

        ar1 = ar1.compile_model()

        rho = 0.85
        sigma = 1.00

        from scipy.stats import norm

        yy = np.asarray(ar1.yy).squeeze()
        yyhat = np.r_[0, rho*yy[:-1]]

        sig = sigma*np.ones_like(yyhat)
        sig[0] = sig[0]/np.sqrt(1.-rho**2)

        z = (yy - yyhat) / sig

        # Kalman Filter
        byhand = np.sum(norm.logpdf(yy, loc=yyhat, scale=sig))
        lik0 = ar1.log_lik([rho, sigma])
        self.assertAlmostEqual(byhand, lik0)


    def test_missing(self):
        relative_loc = 'examples/ar1/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'ar1.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'arma23_sim200.txt')
        ar1 = DSGE.DSGE.read(model_file)
        ar1['__data__']['estimation']['data'] = data_file

        rho, sigma = ar1.p0()

        ar1 = ar1.compile_model()

        res = ar1.kf_everything([rho, sigma], y=np.nan*np.ones_like(ar1.yy))
        ll = res['log_lik'].sum().values[0]

        self.assertAlmostEqual(ll, 0)

        from scipy.stats import norm


        y = ar1.yy.values.copy().squeeze()

        sig = sigma*np.ones_like(y)
        sig[0] = sigma/np.sqrt(1.-rho**2)

        yyhat = np.r_[0, rho*y[:-1]]

        byhand = norm.logpdf(y[:-2], loc=yyhat[:-2], scale=sig[:-2]).sum()
        byhand += norm.logpdf(y[-1], loc=rho**2*y[-3], scale=np.sqrt(1+rho**2)*sig[-1])
        y[-2] = np.nan
        res = ar1.kf_everything([rho, sigma], y=y)
        ll = res['log_lik'].sum().values[0]

        self.assertAlmostEqual(ll, byhand)

    def test_filtered_values(self):
        from dsge.examples import nkmp as dsge

        p0 = dsge.p0()
        model = dsge.compile_model()

        res = model.kf_everything(p0)

        #print(res['filtered_states'])

        self.assertAlmostEqual(0, 0)


    def test_pred(self):
        relative_loc = 'examples/ar1/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'ar1.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'arma23_sim200.txt')
        ar1 = DSGE.DSGE.read(model_file)
        ar1['__data__']['estimation']['data'] = data_file

        rho, sigma = ar1.p0()

        ar1 = ar1.compile_model()

        res = ar1.kf_everything([rho, sigma], y=ar1.yy)

        pred = ar1.pred([rho, sigma], shocks=False,h=5)

        y1 = ar1.yy.iloc[-1].values
        assert_allclose(pred, [rho*y1, rho**2*y1, rho**3*y1, rho**4*y1, rho**5*y1])
