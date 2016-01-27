from __future__ import division

import numpy as np
from slycot import sb03md

def dlyap(TT,RQR):
    n = TT.shape[0]
    U = np.eye(n)
    X,scale,sep,ferr,w =sb03md(n, -RQR, TT.T, U,'D')
    X = TT.dot(X).dot(TT.T) + RQR

    return X,1
