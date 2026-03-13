import unittest
import numpy as np
from scipy.special import digamma, gammaln
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
        s = 1.0
        nu = 6.0
        size = 200000
        samples = self.dist.rvs(s=s, nu=nu, size=size, random_state=0)
        # Check the sample mean and variance against theoretical values
        mean, var = self.dist.stats(s=s, nu=nu, moments='mv')
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        self.assertAlmostEqual(sample_mean, mean, places=2)
        self.assertAlmostEqual(sample_var, var, places=1)

    def test_entropy(self):
        alpha = self.nu / 2.0
        beta = self.nu * self.s**2 / 2.0
        expected_entropy = alpha + gammaln(alpha) + 0.5 * np.log(beta) - np.log(2.0) - (alpha + 0.5) * digamma(alpha)
        entropy_value = self.dist.entropy(self.s, self.nu)
        self.assertAlmostEqual(entropy_value, expected_entropy, places=5)
