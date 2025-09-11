#!/usr/bin/env python3
from numpy import exp, sqrt, pi
from scipy.stats import norm
from scipy.optimize import fsolve

from functools import lru_cache
from scipy.special import ndtr
normpdf = lambda z: 1/sqrt(2*pi)*exp(-0.5*(z**2))
normcdf = ndtr #norm.cdf
norminv = norm.ppf


@lru_cache(maxsize=128)
def get_sigwstar(zwstar, sprd, zeta_spb):
    return fsolve(lambda x: zetaspbfcn(zwstar, x, sprd) - zeta_spb, 0.5)[0]
                  



def zetaspbfcn(z,sigma,sprd):
    zetaratio = zetabomegafcn(z,sigma,sprd)/zetazomegafcn(z,sigma,sprd)
    nk = nkfcn(z,sigma,sprd)
    return -zetaratio/(1-zetaratio)*nk/(1-nk)


def zetabomegafcn(z,sigma,sprd):
    nk = nkfcn(z,sigma,sprd)
    mustar = mufcn(z,sigma,sprd)
    omegastar = omegafcn(z,sigma)
    Gammastar = Gammafcn(z,sigma)
    Gstar = Gfcn(z,sigma)
    dGammadomegastar = dGammadomegafcn(z)
    dGdomegastar = dGdomegafcn(z,sigma)
    d2Gammadomega2star = d2Gammadomega2fcn(z,sigma)
    d2Gdomega2star = d2Gdomega2fcn(z,sigma)
    return omegastar*mustar*nk*(d2Gammadomega2star*dGdomegastar-d2Gdomega2star*dGammadomegastar)/(dGammadomegastar-mustar*dGdomegastar)**2/sprd/(1-Gammastar+dGammadomegastar*(Gammastar-mustar*Gstar)/(dGammadomegastar-mustar*dGdomegastar))


def zetazomegafcn(z,sigma,sprd):
    mustar = mufcn(z,sigma,sprd)
    return omegafcn(z,sigma)*(dGammadomegafcn(z)-mustar*dGdomegafcn(z,sigma))/(Gammafcn(z,sigma)-mustar*Gfcn(z,sigma))


def nkfcn(z,sigma,sprd):
    return 1-(Gammafcn(z,sigma)-mufcn(z,sigma,sprd)*Gfcn(z,sigma))*sprd


def mufcn(z,sigma,sprd):
    return (1-1/sprd)/(dGdomegafcn(z,sigma)/dGammadomegafcn(z)*(1-Gammafcn(z,sigma))+Gfcn(z,sigma))


def omegafcn(z, sigma):
    return exp(sigma*z - 0.5*sigma**2)


def Gfcn(z, sigma):
    return normcdf(z-sigma)


def Gammafcn(z, sigma):
    return omegafcn(z,sigma)*(1-normcdf(z))+normcdf(z-sigma)


def dGdomegafcn(z,sigma):
    return normpdf(z)/sigma


def d2domega2fcn(z, sigma):
    return -z*normpdf(z)/omegafcn(z,sigma)/sigma**2


def dGammadomegafcn(z):
    return 1-normcdf(z)


def d2Gammmadomegafcn(z):
    return -normpdf(z)/omegafcn(z,sigma)/sigma


def d2Gammadomega2fcn(z,sigma):
    return -normpdf(z)/omegafcn(z,sigma)/sigma


def dGdsigmafcn(z,sigma):
    return -z*normpdf(z-sigma)/sigma



def d2Gdomegadsigmafcn(z,sigma):
    return -normpdf(z)*(1-z*(z-sigma))/sigma**2



def dGammadsigmafcn(z,sigma):
    return -normpdf(z-sigma)


def d2Gammadomegadsigmafcn(z,sigma):
    return (z/sigma-1)*normpdf(z)


def d2Gdomega2fcn(z,sigma):
    return -z*normpdf(z)/omegafcn(z,sigma)/sigma**2

if __name__=="__main__":
    print('omegafcn', omegafcn(zwstar, sigma))
    print('Gfcn', Gfcn(zwstar, sigma))
    print('Gammafcn', Gammafcn(zwstar, sigma))
    print('dGdomegafcn', dGdomegafcn(zwstar, sigma))
    print('d2Gdomega2fcn', d2Gdomega2fcn(zwstar, sigma))
    print('dGammadomegafcn', dGammadomegafcn(zwstar))
    print('d2Gammadomega2fcn', d2Gammadomega2fcn(zwstar, sigma))
    print('dGdsigmafcn', dGdsigmafcn(zwstar, sigma))
    print('d2Gdomegadsigmafcn', d2Gdomegadsigmafcn(zwstar, sigma))
    print('dGammadsigmafcn', dGammadsigmafcn(zwstar, sigma))
    print('d2Gammadomegadsigmafcn', d2Gammadomegadsigmafcn(zwstar, sigma))
    print('mufcn', mufcn(zwstar, sigma, sprd))
    print('nkfcn', nkfcn(zwstar, sigma, sprd))
    
