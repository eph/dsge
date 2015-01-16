import numpy as np

class Prior(object):

    def __init__(self, individual_prior):
        self.priors = individual_prior
        self.npara = len(individual_prior)
    def logpdf(self, para):
        ind_density = [x.logpdf(y) for x, y in zip(self.priors, para)]
        return sum(ind_density)
    def rvs(self):
        return np.array([x.rvs() for x in self.priors])
        
        
