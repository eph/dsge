import numpy as np
from scipy.stats import rv_continuous
from scipy.special import gammaln, gammaincc, gammaincinv, gamma

class invgamma_zellner_gen(rv_continuous):
    """
    An inverse gamma random variable as detailed in Zellner (1971). 

    A variable sigma follows an inverse gamma (s, nu) if it has the pdf:

    \begin{align}
    p(\sigma|s,\nu) = \frac{2}{\Gamma(\nu/2)} \left(\frac{\nu s^2}{2}\right)^{\nu/2} \frac{1}{\sigma^{\nu+1}} e^{-\nu s^2 / (2\sigma^2)}. 
    \end{align}
    """
    _support_mask = rv_continuous._open_support_mask

    def _param_info(self):
        return [
            {"name": "s", "value_type": float, "domain": (0, np.inf), "inclusive": (False, False)}, 
            {"name": "nu", "value_type": float, "domain": (0, np.inf), "inclusive": (False, False)}
        ]

    def _shape_info(self):
        return [
            {"name": "s", "value_type": False, "domain": (0, np.inf), "inclusive": (False, False)}, 
            {"name": "nu", "value_type": False, "domain": (0, np.inf), "inclusive": (False, False)}
        ]

    def _logpdf(self, x, s, nu):
        return (np.log(2) - gammaln(nu/2) + (nu/2)*np.log(nu*s**2/2)
                - (nu+1)*np.log(x) - (nu*s**2) / (2*x**2))

    def _pdf(self, x, s, nu):
        return np.exp(self._logpdf(x, s, nu))

    def _cdf(self, x, s, nu):
        return gammaincc(nu/2, nu * s**2 / (2 * x**2))

    def _ppf(self, q, s, nu):
        return np.sqrt(nu * s**2 / (2 * gammaincinv(nu/2, q)))

    def _sf(self, x, s, nu):
        return 1 - self._cdf(x, s, nu)

    def _isf(self, q, s, nu):
        return self._ppf(1 - q, s, nu)

    def _stats(self, s, nu, moments='mvsk'):
        mean = np.inf if nu <= 1 else gamma( (nu - 1) / 2) * np.sqrt(nu * s**2 / 2) / gamma(nu / 2)
        variance = np.inf if nu <= 2 else nu * s**2 / (nu - 2) - mean**2
        skewness = None if nu <= 3 else gamma( (nu - 3) / 2) * (nu * s**2 / 2)**(3/2) / gamma(nu / 2) - 3 * mean * variance - mean**3
        kurtosis = None

        if 's' in moments:
            skewness = None  # Skewness is undefined for this distribution
        if 'k' in moments:
            kurtosis = None  # Kurtosis is undefined for this distribution

        return mean, variance, skewness, kurtosis

    def _entropy(self, s, nu):
        return (nu + 1)/2 * np.log(2) + (nu + 1) * np.log(s) - np.log(gammaln(nu/2)) + (1 - nu)/2 * np.log(nu) + gammaln(nu/2)

    def _rvs(self, s, nu, size=None, random_state=None):
        if size is None:
            shape = 1
        else:
            shape = int(np.prod(size))

        if random_state is None:
            rng = self._random_state
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        rn = rng.standard_normal(size=(shape, int(nu)))

        draws = np.sqrt(nu * s**2 / np.sum(rn**2, axis=-1))
        if size is None:
            return float(draws.reshape(-1)[0])
        return draws.reshape(size)

# Create an instance of the custom distribution
invgamma_zellner = invgamma_zellner_gen(a=0.0, name='invgamma_zellner')

# Test the custom distribution
# pdf_value = invgamma_zellner.pdf(1.0, s=1.0, nu=2.0)  # PDF at x = 1.0
# logpdf_value = invgamma_zellner.logpdf(1.0, s=1.0, nu=2.0)  # log PDF at x = 1.0
# cdf_value = invgamma_zellner.cdf(1.0, s=1.0, nu=2.0)  # CDF at x = 1.0
# ppf_value = invgamma_zellner.ppf(0.5, s=1.0, nu=2.0)  # PPF at q = 0.5
# rvs_values = invgamma_zellner.rvs(s=1.0, nu=2.0, size=5)  # Generate 5 random variates

# pdf_value, logpdf_value, cdf_value, ppf_value, rvs_values
