#!/usr/bin/env python3



########################################################################################################################
#Computes IRFs for FHP-model with investment.
#########################################################################################################################



import numpy as np
import pandas as pd
from dsge.tests import sa
import sys
import importlib

from dsge import read_yaml

########################################################################################################################

#Set parameters of RE model and solve model.

########################################################################################################################

mod_file = 'dsge/examples/fhp/re_cap.yaml'
mod = read_yaml(mod_file)
p0re = mod.p0()
re = mod.compile_model()
[AAre,BBre,unique] = re.solve_LRE(p0re)
TT=11
reirf = re.impulse_response(p0re,TT)

def rational_expectations_irf(p0, TT=11):
    p0re = p0.copy()
    re = mod.compile_model()
    [AAre,BBre,unique] = re.solve_LRE(p0re)

    reirf = re.impulse_response(p0re,TT)
    return reirf


def fhp_irf(p0re, TT=11):
    paramlist = []
    Np = np.size(mod.parameters)
    for i in np.arange(Np):
        paramlist.append(str(mod.parameters[i]))
    params_re = dict(zip(paramlist,p0re))

    #para_func
    beta = 1/(1+params_re['ra']/400)
    rkss = (1/beta)-(1-params_re['delta'])
    sharei = params_re['delta']*params_re['alpha']*beta/(1-beta*(1-params_re['delta']))

    shocklist = []
    ns = np.size(mod.shocks)
    for i in np.arange(ns):
        shocklist.append(str(mod.shocks[i]))


        #Variable names of FHP model
        xnames = ['kp','cc','dp','qq','inv','yy','mc','nr']; nx = len(xnames)
        vvnames = ['vk','vh','vp']; nv = len(vvnames)
        shocknames = ['re','mu','chi','gg','emon']; ns = len(shocknames)
        fhpm0 = {'shocknames': shocknames}
        fhpm0['statenames'] = sa.get_statenames(xnames,vvnames,shocknames)

    #Solve FHP model
    (fhpm0['TT'],fhpm0['RR'],fhpm0['params']) = sa.fhp_companion(p0re,nx,nv,ns,paramlist)

    xirf0 = {}
    for irftype in shocklist:
        #Compute IRF of FHP model
        irfpos = shocklist.index(irftype)  #position of shock to plot
        xirf0[irftype] = sa.compute_irfs(fhpm0,p0re,TT,irfpos)

    return xirf0
