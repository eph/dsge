import numpy as np

#########################################################################################################################St#Functions for Solving FHP-Capital Model
########################################################################################################################

def get_sol_matrices(p0,nx,nv,ns):
    #Returns matrices used to form decision rules for cyclical and trend variables.
    #Inputs: p0plus:  parameters and steady state values

    beta =  1/(1+p0[0]/400)
    dpstar = 1+p0[1]/400
    k = np.int(p0[3])
    sigma = p0[4]
    alpha = p0[5]
    delta = p0[6]
    shareg = p0[7]
    kappa = p0[8]
    phik = p0[9]
    phipi = p0[10]
    phiy = p0[11]
    phipibar = p0[12]
    phiybar = p0[13]
    gammah = p0[14]
    gammapi = p0[15]
    gammak = p0[16]
    rhore = p0[17]
    rhomu = p0[18]
    rhoxi = p0[19]
    rhog = p0[20]
    rhom = p0[21]

    rkss = (1/beta)-(1-delta)
    sharei = delta*alpha*beta/(1-beta*(1-delta))
    sharec = 1-sharei-shareg

    #shock order: S = ['re','mu','xi','gg','mp']
    PP = np.eye(ns)
    PP[0,0] = rhore
    PP[1,1] = rhomu
    PP[2,2] = rhoxi
    PP[3,3] = rhog
    PP[4,4] = rhom

    #cyclical matrices
    #endogenous variable order: Xtil = ['kp','cc','dp','qq','inv','yy','mc','rr']
    alpha_0 = np.zeros([nx,nx]); alpha_1 = np.zeros([nx,nx])

    alpha_0[0,3] = 1; alpha_0[0,7] = 1
    alpha_0[1,1] = 1; alpha_0[1,7] = sigma
    alpha_0[2,2] = 1; alpha_0[2,6] = -kappa
    alpha_0[3,3] = 1; alpha_0[3,4] = -phik
    alpha_0[4,0] = -1/delta; alpha_0[4,4] = 1
    alpha_0[5,1] = -sharec; alpha_0[5,4] = -sharei; alpha_0[5,5] = 1
    alpha_0[6,1] = -1/sigma; alpha_0[6,5] = -alpha/(1-alpha); alpha_0[6,6] = 1
    alpha_0[7,2] = -phipi; alpha_0[7,5] = -phiy; alpha_0[7,7] = 1

    alpha_1[3,0] = -phik; alpha_1[4,0] = -(1-delta)/delta; alpha_1[6,0] = -alpha/(1-alpha)

    beta_0 = np.zeros([nx,ns])
    beta_0[0,0] = 1
    beta_0[1,0] = sigma
    beta_0[3,1] = -1
    beta_0[4,1] = -1
    beta_0[5,3] = shareg
    beta_0[6,2] = 1
    beta_0[7,4] = 1

    alpha_C = alpha_0.copy()
    alpha_C[0,0] = beta*rkss

    alpha_F = np.zeros([nx,nx])
    alpha_F[0,2] = 1; alpha_F[0,3] = beta; alpha_F[0,5] = beta*rkss; alpha_F[0,6] = beta*rkss
    alpha_F[1,1] = 1; alpha_F[1,2] = sigma
    alpha_F[2,2] = beta

    alpha_B = alpha_1.copy()
    beta_s = beta_0.copy()
    beta_s[0,1] = beta*rhomu

    alphabar_0 = alpha_0.copy()
    alphabar_0[7,2] = -phipibar; alphabar_0[7,5] = -phiybar

    alphabar_1 = alpha_1.copy()
    betabar_v = np.zeros([nx,nv])
    betabar_v[0,0] = 1/sigma; betabar_v[0,2] = 1
    betabar_v[1,0] = 1; betabar_v[2,1] = beta

    alphabar_C = alphabar_0.copy()
    alphabar_C[0,0] = beta*rkss

    alphabar_F = alpha_F.copy()
    alphabar_B = alpha_B.copy()

    Ainv = np.linalg.inv(alpha_0)
    Ak = Ainv @ alpha_1
    Bk = Ainv @ beta_0
    Abinv = np.linalg.inv(alphabar_0)
    Abark = Abinv @ alphabar_1
    Bbark = Abinv @ betabar_v

    Cx = np.zeros([nv,nx]); Cs = np.zeros([nv,ns])
    Cx[0,1] = 1; Cx[0,2] = sigma
    Cx[1,2] = 1
    Cx[2,1] = -(1+beta*rkss*(1-alpha)/alpha)/sigma; Cx[2,3] = beta; Cx[2,6] = beta*rkss/alpha
    Cs[2,1] = beta*(1-delta); Cs[2,1] = -beta*rkss*(1-alpha)/alpha

    Gamma = np.zeros([nv,nv])
    Gamma[0,0] = gammah; Gamma[1,1] = gammapi; Gamma[2,2] = gammak

    #note loop goes from jj=1,2,...,k

    for jj in np.arange(1,k+1):
        Ainv = np.linalg.inv(alpha_C - alpha_F @ Ak)
        Abinv = np.linalg.inv(alphabar_C - alphabar_F @ Abark)
        Ak = Ainv @ alpha_B
        Bk = Ainv @ (alpha_F @ Bk @ PP + beta_s)
        Abark = Abinv @ alphabar_B
        Bbark = Abinv @ alphabar_F @ Bbark

    return(Ak,Bk,Abark,Bbark,Cx,Cs,Gamma,PP)

def get_statenames(xynames,vvnames,shocknames):

    xytil = xynames.copy()
    xybar = xynames.copy()
    for i in range(len(xytil)):
        xytil[i] = xytil[i]+'til'
        xybar[i] = xybar[i]+'bar'
    statenames = xynames+xytil+xybar+vvnames+shocknames
    return(statenames)

def fhp_companion(p0,nx,nv,ns,paramlist):
    #Put model solution in companion form

    beta =  1/(1+p0[0]/400)
    sigma = p0[4]
    delta = p0[6]
    phik = p0[9]
    gammah = p0[14]
    gammap = p0[15]
    gammak = p0[16]
    rkss = (1/beta)-(1-delta)

    params0 = dict(zip(paramlist,p0))

    (Ak,Bk,Abark,Bbark,Cx,Cs,Gamma,PP) = get_sol_matrices(p0,nx,nv,ns)

    #Put solution in companion form:  Y = TT Y(-1) + RR ee
    # Y = (X,Xtil,Xbar,V,s)

    TT = np.zeros([3*nx+nv+ns,3*nx+nv+ns])
    RR = np.zeros([3*nx+nv+ns,ns])

    print('---------')
    print((Gamma).round(3))
    #X and Y
    eyev = np.eye(nv)
    TT[:nx,:nx] = Bbark @ Gamma @ Cx; TT[:nx,nx:2*nx] = Ak; TT[:nx,2*nx:3*nx] = Abark
    TT[:nx,3*nx:3*nx+nv] = Bbark @ (eyev - Gamma); TT[:nx,3*nx+nv:] = Bk @ PP + Bbark @ Gamma @ Cs
    TT[nx:2*nx,nx:2*nx] = Ak; TT[nx:2*nx,3*nx+nv:] = Bk @ PP
    TT[2*nx:3*nx,:nx] = Bbark @ Gamma @ Cx; TT[2*nx:3*nx,2*nx:3*nx] = Abark
    TT[2*nx:3*nx,3*nx:3*nx+nv] = Bbark @ (eyev - Gamma); TT[2*nx:3*nx,3*nx+nv:] = Bbark @ Gamma @ Cs
    TT[3*nx:3*nx+nv,:nx] = Gamma @ Cx; TT[3*nx:3*nx+nv,3*nx:3*nx+nv] = eyev - Gamma;  TT[3*nx:3*nx+nv,3*nx+nv:] = Gamma @ Cs
    TT[3*nx+nv:,3*nx+nv:] = PP

    RR[:nx,:] = Bk; RR[nx:2*nx,:] = Bk; RR[3*nx+nv:,:] = np.eye(ns)

    return(TT,RR,params0)

def compute_irfs(fhpmod,p0,capt,irfpos):
    import pandas as pd

    ncol = np.shape(fhpmod['statenames'])[0]
    xirf0 = pd.DataFrame(np.zeros([capt,ncol]),columns=fhpmod['statenames'])

    ndf = len(fhpmod['statenames'])
    ns = len(fhpmod['shocknames'])
    YYm1 = np.zeros(ndf)
    innov = np.zeros([ns,capt])
    innov[irfpos,0] = p0[22+irfpos]
    for tt in np.arange(0,capt):
        YY = fhpmod['TT'].dot(YYm1) + fhpmod['RR'].dot(innov[:,tt])
        xirf0.loc[tt,fhpmod['statenames']] = YY
        YYm1 = YY
    return(xirf0)
