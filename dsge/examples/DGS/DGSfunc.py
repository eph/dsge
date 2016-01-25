from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
def d2Gammadomegadsigmafcn(z,sigma):
    f = (z/sigma-1)*norm.pdf(z);
    return f

def dGammadsigmafcn(z,sigma):
    f = -norm.pdf(z-sigma);
    return f


def zetaspbfcn(z,sigma,sprd):
    zetaratio = zetabomegafcn(z,sigma,sprd)/zetazomegafcn(z,sigma,sprd);
    nk = nkfcn(z,sigma,sprd);
    f = -zetaratio/(1-zetaratio)*nk/(1-nk);
    return f
        
def zetabomegafcn(z,sigma,sprd):
    nk = nkfcn(z,sigma,sprd);
    mustar = mufcn(z,sigma,sprd);
    omegastar = omegafcn(z,sigma);
    Gammastar = Gammafcn(z,sigma);
    Gstar = Gfcn(z,sigma);
    dGammadomegastar = dGammadomegafcn(z);
    dGdomegastar = dGdomegafcn(z,sigma);
    d2Gammadomega2star = d2Gammadomega2fcn(z,sigma);
    d2Gdomega2star = d2Gdomega2fcn(z,sigma);
    f = (omegastar*mustar*nk*(d2Gammadomega2star*dGdomegastar-d2Gdomega2star*dGammadomegastar)/
         (dGammadomegastar-mustar*dGdomegastar)**2/sprd/
         (1-Gammastar+dGammadomegastar*(Gammastar-mustar*Gstar)/(dGammadomegastar-mustar*dGdomegastar)))
    return f

def zetazomegafcn(z,sigma,sprd):
    mustar = mufcn(z,sigma,sprd);
    f = (omegafcn(z,sigma)*(dGammadomegafcn(z)-mustar*dGdomegafcn(z,sigma))/
    (Gammafcn(z,sigma)-mustar*Gfcn(z,sigma)));
    return f

def nkfcn(z,sigma,sprd):
    f = 1-(Gammafcn(z,sigma)-mufcn(z,sigma,sprd)*Gfcn(z,sigma))*sprd;
    return f

def mufcn(z,sigma,sprd):
    f = (1-1/sprd)/(dGdomegafcn(z,sigma)/dGammadomegafcn(z)*(1-Gammafcn(z,sigma))+Gfcn(z,sigma));
    return f

def omegafcn(z,sigma):
    f = np.exp(sigma*z-1/2*sigma**2);
    return f

def Gfcn(z,sigma):
    f = norm.cdf(z-sigma);
    return f

def Gammafcn(z,sigma):
    f = omegafcn(z,sigma)*(1-norm.cdf(z))+norm.cdf(z-sigma);
    return f

def dGdomegafcn(z,sigma):
    f = norm.pdf(z)/sigma;
    return f

def d2Gdomega2fcn(z,sigma):
    f = -z*norm.pdf(z)/omegafcn(z,sigma)/sigma**2;
    return f

def dGammadomegafcn(z):
    f = 1-norm.cdf(z);
    return f

def d2Gammadomega2fcn(z,sigma):
    f = -norm.pdf(z)/omegafcn(z,sigma)/sigma;
    return f

def dGdsigmafcn(z,sigma):
    f = -z*norm.pdf(z-sigma)/sigma;
    return f

def d2Gdomegadsigmafcn(z,sigma):
    f = -norm.pdf(z)*(1-z*(z-sigma))/sigma**2;
    return f

norminv = norm.ppf
from scipy.optimize import fsolve
def get_sigwstar(zwstar, sprd, zeta_spb):
    return fsolve(lambda x: zetaspbfcn(zwstar, x, sprd)-zeta_spb, 0.5)