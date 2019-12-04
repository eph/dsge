from __future__ import division
import numpy as np

from scipy.special import gammaln
import scipy.stats as rv

class InvGamma(object):
    """
    An inverse gamma random variable as detailed in Zellner (1971). 

    A variable sigma follows an inverse gamma (s, nu) if it has the pdf:

    \begin{align}
    p(\sigma|s,\nu) = \frac{2}{\Gamma(\nu/2)} \left(\frac{\nu s^2}{2}\right)^{\nu/2} \frac{1}{\sigma^{\nu+1}} e^{-\nu s^2 / (2\sigma^2)}. 
    \end{align}

    """
    name = 'inv_gamma'

    def __init__(self, s, nu):

        self.a = s
        self.b = nu

    def logpdf(self, x):
        a = self.a
        b = self.b
        if x < 0:
            return -1000000000000

        lpdf = (np.log(2) - gammaln(b/2) + b/2*np.log(b*a**2/2)
                -(b+1)/2*np.log(x**2) - b*a**2/(2*x**2))
        return lpdf
        
    
    def rvs(self):
        rn = rv.norm.rvs(size=(int(self.b), 1))
        return np.sqrt(self.b*self.a**2 / np.sum(rn**2, 0))


class InvGamma1(object):
    """Inverse Gamma 1 distribution
    X ~ IG1(s, nu) if X = sqrt(Y), where Y ~IG2(s, nu) with Y = INV(Z), Z~Gamma(nu/2, 2/s)
    """
    def __init__(self, s, nu):
    
        self.s = s
        self.nu = nu

    def rvs(self):
        return np.sqrt(1.0/rv.gamma.rvs(self.nu/2.0, scale=2.0/self.s))


    def logpdf(self, x):
        pass
