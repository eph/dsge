from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import DSGE

import pkg_resources

class TestFilter(TestCase):

    def test_sw_lik(self):
        relative_loc = 'examples/sw/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'sw.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'YY.txt')

        sw = DSGE.DSGE.read(model_file)
        sw['__data__']['estimation']['data']['file'] = data_file

        sw = sw.compile_model()

        # p0 = [0.108643019053367,
        #       0.446238477394778,
        #       -1.40755240925405,
        #       0.309662450888992,
        #       0.761200080182307,
        #       1.11460719207234,
        #       2.80635604115762,
        #       0.599229766348742,
        #       0.608129789293928,
        #       1.54733288002788,
        #       0.409901175503114,
        #       0.552266892246550,
        #       0.438356013728308,
        #       0.409126173154452,
        #       2.17194728385529,
        #       0.770413749895747,
        #       0.107772208246328,
        #       0.139955785335781,
        #       0.603142571054213,
        #       0.345511967851210,
        #       0.330248138400818,
        #       0.464220851694531,
        #       0.268423352222895,
        #       0.107948523100408,
        #       0.560176102788254,
        #       0.704591287775303,
        #       0.725055425970607,
        #       0.280847730707524,
        #       5.627479580922513E-002,
        #       0.140457960025464,
        #       0.287737282637854,
        #       9.992532674618690E-002,
        #       0.181882631592647,
        #       0.106318562611565,
        #       0.161003018496727]

        p0 = [0.1657,0.7869,0.5509,0.1901,1.3333,1.6064,5.7606,0.72,0.7,1.9,0.65,0.57,0.3,0.5462,2.0443,0.8103,0.0882,0.2247,0.9577,0.2194,0.9767,0.7113,0.1479,0.8895,0.9688,0.5,0.72,0.85,0.4582,0.24,0.5291,0.4526,0.2449,0.141,0.2446]
        lik0 = sw.log_lik(p0)

        self.assertAlmostEqual(-829.7412615500879, lik0, places=6)

    def test_sw_qlik(self):
        relative_loc = 'examples/sw/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'sw.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'YY.txt')

        sw = DSGE.DSGE.read(model_file)
        sw['__data__']['estimation']['data']['file'] = data_file

        sw = sw.compile_model()

        p0 = [0.108643019053367,
              0.446238477394778,
              -1.40755240925405,
              0.309662450888992,
              0.761200080182307,
              1.11460719207234,
              2.80635604115762,
              0.599229766348742,
              0.608129789293928,
              1.54733288002788,
              0.409901175503114,
              0.552266892246550,
              0.438356013728308,
              0.409126173154452,
              2.17194728385529,
              0.770413749895747,
              0.107772208246328,
              0.139955785335781,
              0.603142571054213,
              0.345511967851210,
              0.330248138400818,
              0.464220851694531,
              0.268423352222895,
              0.107948523100408,
              0.560176102788254,
              0.704591287775303,
              0.725055425970607,
              0.280847730707524,
              5.627479580922513E-002,
              0.140457960025464,
              0.287737282637854,
              9.992532674618690E-002,
              0.181882631592647,
              0.106318562611565,
              0.161003018496727]

        lik0 = sw.log_quasilik_hstep(p0, h=1)

        #self.assertAlmostEqual(-108607.442008, lik0, places=6)

        #lik0 = sw.log_quasilik_hstep(p0, h=4)


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


        h = 2
        yyhat = np.r_[0, rho*(1+rho)/2*yy[:-1]]
        yymean = (yy + np.r_[yy[1:], 0])/2
        sig = np.sqrt( 2**-2*((1+rho)**2 + 1**2)) * sigma * np.ones_like(yyhat)
        sig[0] = np.sqrt( (rho*(1+rho)/2)**2/(1-rho**2) +((1+rho)/2)**2 + (1/2)**2) * sigma
        byhand = norm.logpdf(yymean[:-(h-1)], loc=yyhat[:-(h-1)], scale=sig[:-(h-1)]).sum()

        lik0 = ar1.log_quasilik_hstep([rho, sigma], h=2)
        print byhand, lik0
        self.assertAlmostEqual(byhand, lik0)

        h = 3
        hsum = (rho**np.arange(1, h+1)).mean()
        yyhat = np.r_[0, hsum*yy[:-1]]
        yymean = np.convolve(np.repeat(1.0, h)/h, yy, 'valid')

        sigscale = h**-2*np.array([((rho**np.arange(i)).sum()**2) for i in range(1, h+1)]).sum()
        sig = np.sqrt(sigscale)*sigma*np.ones_like(yyhat)
        sig[0] = np.sqrt(h**-2*(rho**np.arange(1, h+1)).sum()**2/(1-rho**2) + sigscale)*sigma

        byhand = norm.logpdf(yymean, loc=yyhat[:-(h-1)], scale=sig[:-(h-1)]).sum()
        lik0 = ar1.log_quasilik_hstep([rho, sigma], h=h)

        print byhand, lik0
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

    def test_h_step(self):
        pass

    def test_filtered_values(self):
        relative_loc = 'examples/nkmp/'
        model_file = pkg_resources.resource_filename('dsge', relative_loc+'nkmp.yaml')
        data_file = pkg_resources.resource_filename('dsge', relative_loc+'us.txt')

        dsge = DSGE.DSGE.read(model_file)
        dsge['__data__']['estimation']['data']['file'] = data_file

        p0 = dsge.p0()
        model = dsge.compile_model()

        res = model.kf_everything(p0)

        print res['filtered_states']

        self.assertAlmostEqual(0, 0)
