import numpy as np

from numba import jit

@jit('f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])', nopython=True)
def chand_recursion(y, TT, RR, QQ, DD, ZZ, HH, P0):
    nobs, ny = y.shape
    ns = TT.shape[0]

    Pt = P0

    loglh = 0.0
    At = np.zeros(shape=(ns))


    Ft = ZZ @ Pt @ ZZ.T + HH
    Ft = 0.5 * (Ft + Ft.T)
    iFt = np.linalg.inv(Ft)

    St = TT @ Pt @ ZZ.T
    Mt = -iFt
    Kt = St @ iFt

    for i in range(nobs):
        yhat = ZZ @ At + DD.flatten()
        nut = y[i] - yhat

        dFt = np.log(np.linalg.det(Ft))
        iFtnut = np.linalg.solve(Ft, nut)

        loglh = loglh - 0.5*ny*np.log(2*np.pi) - 0.5*dFt - 0.5*np.dot(nut, iFtnut)

        At = TT@At + Kt @ nut.T
        
        ZZSt = ZZ@St;
        MSpZp = Mt@(ZZSt.T);
        TTSt = TT@St;

        Ft1  = Ft + ZZSt@MSpZp;         # F_{t+1}
        Ft1  = 0.5*(Ft1+Ft1.T);           
        iFt1 = np.linalg.inv(Ft1);
        
        Kt = (Kt@Ft + TTSt@MSpZp)@iFt1; # K_{t+1}
        St = TTSt - Kt@ZZSt;            # S_{t+1}
        Mt = Mt + MSpZp@iFt@MSpZp.T;     # M_{t+1}
        Mt = 0.5*(Mt + Mt.T);
        Ft = Ft1;
        iFt = iFt1;

    return loglh



@jit('f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])', nopython=True)
def kalman_filter(y, TT, RR, QQ, DD, ZZ, HH, P0):

    #y = np.asarray(y)
    nobs, ny = y.shape
    ns = TT.shape[0]

    RQR = np.dot(np.dot(RR, QQ), RR.T)
    Pt = P0

    loglh = 0.0
    AA = np.zeros(shape=(ns))
    for i in range(nobs):

        not_missing = ~np.isnan(y[i])
        nact = not_missing.sum()
        yhat = ZZ[not_missing,:] @ AA + DD.flatten()[not_missing]

        nut = y[i][not_missing] - yhat

        Ft = ZZ[not_missing,:] @ Pt @ ZZ[not_missing,:].T + HH[not_missing,:][:,not_missing]
        Ft = 0.5 * (Ft + Ft.T)

        dFt = np.log(np.linalg.det(Ft))
        iFtnut = np.linalg.solve(Ft, nut)

        loglh = loglh - 0.5*nact*np.log(2*np.pi) - 0.5*dFt - 0.5*np.dot(nut, iFtnut)
 
        TTPt = TT @ Pt
        Kt = TTPt @ ZZ[not_missing,:].T

        AA = TT @ AA + Kt @ iFtnut
        Pt = TTPt @ TT.T - Kt @ np.linalg.solve(Ft, Kt.T) + RQR

    return loglh
