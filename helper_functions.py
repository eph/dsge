import numpy as np
def EXP(j):
    return lambda x: E(j, x)

def E(j, x):
    if len(x.atoms()) > 1:

        for xa in x.atoms():
            if isinstance(xa, Variable):
                x = x.subs({xa:E(j, xa)})
        return x
    else:
        if isinstance(x, Parameter):
            return x
        if isinstance(x, Variable):
            return type(x)(x.name, date=x.date, exp_date=j)

def ASUM(x, d):
    return x

def cholpsd(x):
    n = x.shape[0]
    y = x.copy()
    for k in np.arange(0, n):
        if y[k, k]>0:
            y[k, k] = np.sqrt(y[k, k])
            if k < n-1:
                y[k+1:, k] = y[k+1:, k]/y[k, k]
                y[k, k+1:] = 0
                for j in np.arange(k+1, n):
                    y[j:, j] = y[j:, j] - y[j:, k]*y[j, k]

    return y
