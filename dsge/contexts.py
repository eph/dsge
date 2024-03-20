import numpy as np
import sympy 

from .symbols import Parameter, Variable

function_context = {
        'exp': sympy.exp,
        'log': sympy.log,
        'betacdf': sympy.Function('betacdf'),}

numeric_context = {'ImmutableDenseMatrix': np.array}


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

si_context = {'EXP': EXP, 'E': E, 'SUM': ASUM, 'oo': sympy.oo, 'inf': sympy.oo}
