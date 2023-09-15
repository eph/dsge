#!/usr/bin/env python3



########################################################################################################################

#Computes IRFs for FHP-model with investment.

#########################################################################################################################



import numpy as np

import pandas as pd

from dsge.tests import sa

import sys

import importlib
from dsge.DSGE import DSGE



########################################################################################################################

#Set parameters of RE model and solve model.

########################################################################################################################


mod_file = 'dsge/examples/fhp/re_cap.yaml'
mod = DSGE.read(mod_file)
p0re = mod.p0()

#change some model parameters from whats in the yaml file
kk = 12000

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

re = mod.compile_model()
[AAre,BBre,unique] = re.solve_LRE(p0re)

######################################################################################
#Solve FHP model at k very large.  Compute IRFs and compare to RE solution.
######################################################################################

#Variable names of FHP model
xnames = ['kp','cc','dp','qq','inv','yy','mc','nr']; nx = len(xnames)
vvnames = ['vk','vh','vp']; nv = len(vvnames)
shocknames = ['re','mu','chi','gg','emon']; ns = len(shocknames)
fhpm0 = {'shocknames': shocknames}
fhpm0['statenames'] = sa.get_statenames(xnames,vvnames,shocknames)

#Solve FHP model
(fhpm0['TT'],fhpm0['RR'],fhpm0['params']) = sa.fhp_companion(p0re,nx,nv,ns,paramlist)

#Compute IRFs of RE model
TT = 20  #length of irfs
reirf = re.impulse_response(p0re,TT)
irftype = 'epg'
xirf_re = 100*reirf[irftype].copy()

#Compute IRF of FHP model
irfpos = shocklist.index(irftype)  #position of shock to plot
xirf0 = 100*sa.compute_irfs(fhpm0,p0re,TT,irfpos)
#fhp_plt.plot_irf_fhp_compare(xirf0,xirf_re,'FHP','RE',irfpos)

######################################################################################
# #Solve FHP model at k low.  Compute IRFs and compare to FHP with very large k.
######################################################################################

p1 = p0re.copy()
kk = 1
p1[3] = kk
fhpm1 = fhpm0.copy()
(fhpm1['TT'],fhpm1['RR'],fhpm1['params']) = sa.fhp_companion(p1,nx,nv,ns,paramlist)
xirf1 = 100*sa.compute_irfs(fhpm1,p1,TT,irfpos)

#fhp_plt.plot_irf_fhp_compare(xirf1,xirf_re,'FHP k low','RE',irfpos)
#fhp_plt.plot_irf_decomp(xirf1,irfpos)
