import numpy as np

from scipy.stats import beta, norm, uniform, gamma
from .OtherPriors import InvGamma

def construct_prior(prior_list, parameters):

    prior = []
    for par in parameters:
        prior_spec = prior_list[par]

        ptype = prior_spec[0]
        pmean = prior_spec[1]
        pstdd = prior_spec[2]
        if ptype == 'beta':
            a = (1 - pmean) * pmean**2 / pstdd**2 - pmean
            b = a * (1 / pmean - 1)
            pr = beta(a, b)
            pr.name = 'beta'
            prior.append(pr)
        if ptype == 'gamma':
            b = pstdd**2 / pmean
            a = pmean / b
            pr = gamma(a, scale=b)
            pr.name = 'gamma'
            prior.append(pr)
        if ptype == 'normal':
            a = pmean
            b = pstdd
            pr = norm(loc=a, scale=b)
            pr.name = 'norm'
            prior.append(pr)
        if ptype == 'inv_gamma':
            a = pmean
            b = pstdd
            prior.append(InvGamma(a, b))
        if ptype == 'uniform':
            a, b = pmean, pstdd
            pr = uniform(loc=a, scale=(b - a))
            pr.name = 'uniform'
            prior.append(pr)

    return prior


class Prior(object):
    def __init__(self, individual_prior):
        self.priors = individual_prior
        if self.priors is None:
            self.npara = 0
        else:
            self.npara = len(individual_prior)

    def logpdf(self, para):
        if self.priors is None:
            return None

        ind_density = [x.logpdf(y) for x, y in zip(self.priors, para)]
        ldens = np.sum(ind_density)
        if ldens == -np.inf:
            ldens = -100000000000.0
        return ldens

    def rvs(self, size=None):
        if self.priors is None:
            return None
        if size == None:
            return np.array([x.rvs() for x in self.priors])
        else:
            return np.array([[x.rvs() for x in self.priors] for _ in range(size)])
