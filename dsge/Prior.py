import numpy as np

from typing import Optional, Union

from numpy.random import Generator, RandomState
from scipy.stats import beta, norm, uniform, gamma
from .OtherPriors import invgamma_zellner

pdict = {"gamma": 1, "beta": 2, "norm": 3, "invgamma_zellner": 4, "uniform": 5}
def construct_prior(prior_list, parameters):

    prior = []
    for par in parameters:
        prior_spec = prior_list[par]

        ptype = prior_spec[0]
        pmean = prior_spec[1]
        pstdd = prior_spec[2]
        if ptype == "beta":
            a = (1 - pmean) * pmean ** 2 / pstdd ** 2 - pmean
            b = a * (1 / pmean - 1)
            pr = beta(a, b)
            pr.name = "beta"
            prior.append(pr)
        if ptype == "gamma":
            b = pstdd ** 2 / pmean
            a = pmean / b
            pr = gamma(a, scale=b)
            pr.name = "gamma"
            prior.append(pr)
        if ptype == "normal":
            a = pmean
            b = pstdd
            pr = norm(loc=a, scale=b)
            pr.name = "norm"
            prior.append(pr)
        if ptype == "inv_gamma":
            a = pmean
            b = pstdd
            prior.append(invgamma_zellner(a, b))
        if ptype == "uniform":
            a, b = pmean, pstdd
            pr = uniform(loc=a, scale=(b - a))
            pr.name = "uniform"
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

    def rvs(self, size: Optional[int] = None, random_state: Optional[Union[int, Generator, RandomState]] = None):
        if self.priors is None:
            return None

        if size is not None and not isinstance(size, int):
            raise TypeError("size must be an integer or None")

        rng: Optional[Union[int, Generator, RandomState]]
        if isinstance(random_state, Generator) or isinstance(random_state, RandomState):
            rng = random_state
        elif random_state is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(random_state)

        def _draw(dist, draw_size: Optional[int]):
            sample = dist.rvs(size=draw_size, random_state=rng)
            arr = np.asarray(sample)
            if draw_size is None:
                if arr.ndim == 0:
                    return float(arr)
                if arr.size == 1:
                    return float(arr.reshape(-1)[0])
                raise ValueError(
                    f"Prior draw for distribution '{getattr(dist, 'name', type(dist).__name__)}' "
                    f"returned shape {arr.shape}, expected scalar."
                )
            return arr.reshape(draw_size)

        if size is None:
            return np.array([_draw(dist, None) for dist in self.priors], dtype=float)

        draws = np.empty((size, len(self.priors)), dtype=float)
        for idx, dist in enumerate(self.priors):
            draws[:, idx] = _draw(dist, size)
        return draws

    def fortran_prior(self):
        def return_stats(dist):
            if dist.name == "uniform":
                return (
                    pdict[dist.name],
                    dist.kwds["loc"],
                    dist.kwds["loc"] + dist.kwds["scale"],
                    0,
                    0,
                 )
            elif dist.name == "invgamma_zellner":
                return pdict[dist.name], dist.a, dist.b, 0, 0
            else:
                return pdict[dist.name], dist.stats()[0], np.sqrt(dist.stats()[1]), 0, 0

        return np.array([return_stats(x) for x in self.priors])
