import unittest
import numpy as np
from scipy.special import gammaln
from dsge.OtherPriors import invgamma_zellner

class TestInvGammaZellner(unittest.TestCase):
    def setUp(self):
        self.dist = invgamma_zellner
        self.s = 1.0
        self.nu = 2

    def test_pdf(self):
        x = 1.0
        expected_pdf = 0.7357588823428847
        pdf_value = self.dist.pdf(x, s=self.s, nu=self.nu)
        self.assertAlmostEqual(pdf_value, expected_pdf, places=5)

    def test_logpdf(self):
        x = 1.0
        expected_logpdf = -0.3068528194400547
        logpdf_value = self.dist.logpdf(x, s=self.s, nu=self.nu)
        self.assertAlmostEqual(logpdf_value, expected_logpdf, places=5)

    def test_cdf(self):
        x = 1.0
        expected_cdf = 0.36787944117144245
        cdf_value = self.dist.cdf(x, s=self.s, nu=self.nu)
        self.assertAlmostEqual(cdf_value, expected_cdf, places=5)

    def test_ppf(self):
        q = 0.5
        expected_ppf = 1.2011224087864496
        ppf_value = self.dist.ppf(q, s=self.s, nu=self.nu)
        self.assertAlmostEqual(ppf_value, expected_ppf, places=5)


    def test_rvs(self):
        size = 50000000
        samples = self.dist.rvs(s=self.s, nu=self.nu, size=size)
        # Check the sample mean and variance against theoretical values
        mean, var, _, _ = self.dist.stats(s=self.s, nu=self.nu, moments='mvsk')
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        print(mean, sample_mean, var, sample_var)
        if np.isfinite(mean):
            self.assertAlmostEqual(sample_mean, mean, places=2)
        if np.isfinite(var):
            self.assertAlmostEqual(sample_var, var, places=1)

    def test_entropy(self):
        expected_entropy = (self.nu + 1)/2 * np.log(2) + (self.nu + 1) * np.log(self.s) - np.log(gammaln(self.nu/2)) + (1 - self.nu)/2 * np.log(self.nu) + gammaln(self.nu/2)
        entropy_value = self.dist.entropy(self.s, self.nu)
        self.assertAlmostEqual(entropy_value, expected_entropy, places=5)
