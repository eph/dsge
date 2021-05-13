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
