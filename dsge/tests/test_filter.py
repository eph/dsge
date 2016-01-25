from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import DSGE

class TestFilter(TestCase):

    def test_sw_lik(self):
        
        sw = DSGE.DSGE.read('/mq/edprojects/projects/direct-dsge-estimation/code/models/sw/sw.yaml')
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

        lik0 = sw.log_lik(p0)

        self.assertAlmostEqual(-108607.442008, lik0, places=6)

    def test_sw_qlik(self):
        sw = DSGE.DSGE.read('/mq/edprojects/projects/direct-dsge-estimation/code/models/sw/sw.yaml')
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

        ar1 = DSGE.DSGE.read('/mq/home/m1eph00/python-repo/dsge/dsge/examples/ar1/ar1.yaml')
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

        # h = 2
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

        
        ar1 = DSGE.DSGE.read('/mq/home/m1eph00/python-repo/dsge/dsge/examples/ar1/ar1.yaml')
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
        
        dsge = DSGE.DSGE.read('/mq/home/m1eph00/python-repo/dsge/dsge/examples/nkmp/nkmp.yaml')
        p0 = dsge.p0()
        model = dsge.compile_model()

        res = model.kf_everything(p0)

        print res['filtered_states']

        self.assertAlmostEqual(0, 1)