import numpy as np


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
