import SIDSGE
import yaml
import sympy
from sympy import Function
import numpy as np
from symbols import Variable, Parameter, Shock, Equation
from symbols import timeshift as TS
from sympy.matrices import Matrix
from collections import OrderedDict
import re

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

def read(fstr):
    f = open(fstr)
    txt = f.read()
    f.close()

    
    txt = txt.replace('^', '**')
    txt = txt.replace(';', '')
    txt = re.sub(r"@ ?\n", " ", txt)

    model_yaml = yaml.load(txt)

    dec = model_yaml['declarations']
    cal = model_yaml['calibration']

    name = dec['name']

    var_ordering = [Variable(v) for v in dec['variables']]
    par_ordering = [Parameter(v) for v in dec['parameters']]
    shk_ordering = [Shock(v) for v in dec['shocks']]
    exo_ordering = [Variable('exo_'+v) for v in dec['shocks']]

    exo_subs_dict = zip(shk_ordering, exo_ordering)

    if 'para_func' in dec:
        other_para   = [Parameter(v) for v in dec['para_func']]
    else:
        other_para = []

    if 'observables' in dec:
        observables = [Variable(v) for v in dec['observables']]
        obs_equations = model_yaml['equations']['observables']
        
    else:
        observables = []
        obs_equations = dict()

    index = [Variable(j) for j in dec['index']]
    
    context = [(s.name,s) for s in var_ordering + par_ordering + index + shk_ordering + other_para];
    context = dict(context)


    context['TS'] = TS
    context['EXP'] = EXP

    context['inf'] = sympy.oo
    for f in [sympy.log, sympy.exp,
              sympy.sin, sympy.cos, sympy.tan,
              sympy.asin, sympy.acos, sympy.atan,
              sympy.sinh, sympy.cosh, sympy.tanh,
              sympy.pi, sympy.sign]:
        context[str(f)] = f


    context['SUM'] = sympy.Sum
    rcontext = context.copy()
    rcontext['SUM'] = ASUM

    raw_equations = model_yaml['equations']['model']
    equations = []
    sum_rem_equations = []


  
    for obs in obs_equations.iteritems():
        obs_equations[obs[0]] = eval(obs[1], context)

    for eq in raw_equations:
        if '=' in eq:
            lhs, rhs = str.split(eq, '=')
        else:
            lhs = eq
            rhs = '0'

        lhs = eval(lhs, rcontext)
        rhs = eval(rhs, rcontext)

        sum_rem_equations.append(Equation(lhs, rhs).subs(exo_subs_dict))


    for eq in raw_equations:
        if '=' in eq:
            lhs, rhs = str.split(eq, '=')
        else:
            lhs = eq
            rhs = '0'

        lhs = eval(lhs, context)
        rhs = eval(rhs, context)

        equations.append(Equation(lhs, rhs).subs(exo_subs_dict))

    import itertools
    it = itertools.chain.from_iterable

    all_vars = [list(equ.atoms(Variable)) for equ in equations]
    max_lead_endo = dict.fromkeys(var_ordering)
    max_lag_endo = dict.fromkeys(var_ordering)

    for v in var_ordering:
        max_lead_endo[v] = max([i.date for i in it(all_vars) if i.name == v.name])
        max_lag_endo[v] = min([i.date for i in it(all_vars) if i.name == v.name])


    subs_dict = dict()
    old_var = var_ordering[:]
    for v in old_var:

        # lags 
        for i in np.arange(2, abs(max_lag_endo[v])+1):
            # for lag l need to add l-1 variable
            var_l = Variable(v.name + "_LAG" + str(i-1))

            if i == 2:
                var_l_1 = Variable(v.name, date=-1)
            else:
                var_l_1 = Variable(v.name + "_LAG" + str(i-2), date=-1)

            subs_dict[Variable(v.name, date=-i)] = var_l(-1)
            var_ordering.append(var_l)
            equations.append(Equation(var_l, var_l_1))
            sum_rem_equations.append(Equation(var_l, var_l_1))


    equations = [eq.subs(subs_dict) for eq in equations]
    sum_rem_equations = [eq.subs(subs_dict) for eq in sum_rem_equations]

    exo_equations = []
    #raw_equations = model_yaml['equations']['exogenous']
    for exo, shk in exo_subs_dict:
        exo_equations.append(Equation(exo, shk))

    
    calibration = model_yaml['calibration']['parameters']
    p = calibration


    if 'para_func' in model_yaml['calibration']:
        para_func = model_yaml['calibration']['para_func']
    else:
        para_func = []

    func_dict = OrderedDict()
        
    opr = other_para
    
    #for fp in opr:
    #    func_dict[fp] =#eval(str(para_func[str(fp)]), context)


    shk_cal = model_yaml['calibration']['covariances']

    QQ = sympy.zeros(len(shk_ordering), len(shk_ordering))
    
    for key, val in shk_cal.iteritems():
        
        shocks = key.split(",")

        if len(shocks)==1:
            shocks.append(shocks[0])

        if len(shocks)==2:
            shocki = Shock(shocks[0].strip())
            shockj = Shock(shocks[1].strip())
                
            indi = shk_ordering.index(shocki)
            indj = shk_ordering.index(shockj)

            QQ[indi, indj] = eval(val, context)
            QQ[indj, indi] = QQ[indi, indj]

    
    model_dict = {'var_ordering': var_ordering, 
                  'exo_ordering': exo_ordering, 
                  'par_ordering': par_ordering, 
                  'shk_ordering': shk_ordering, 
                  'index': index, 
                  'equations': equations, 
                  'exo_equations': exo_equations, 
                  'sequations': sum_rem_equations, 
                  'calibration': calibration, 
                  'p': calibration, 
                  'QQ': QQ, 
                  'para_func': para_func, 
                  'other_para': other_para, 
                  'observables': observables, 
                  'obs_equations': obs_equations, 
                  'name': name, 
                  '__data__': model_yaml, 
                  'para_func_d': func_dict
              }

    model = SIDSGE.SIDSGE(**model_dict)
    return model

def iteritems(d):
    return zip(d.keys(), d.values())


if __name__=="__main__":
    import sys
    
    if len(sys.argv) >= 2:
        mfile = sys.argv[1]
        model = read(mfile)
        if len(sys.argv) == 3:
            odir = sys.argv[2]
            
        model.write_matlab_model(odir)
        

 

