#!/usr/bin/env python3
"""
SIDSGE - Sticky Information DSGE model implementation.

This module provides classes for working with Sticky Information Dynamic 
Stochastic General Equilibrium models, which include information
rigidities with distributed lags in expectations.
"""

import numpy as np
import sympy
from typing import Dict, Any

from .Base import Base
from .symbols import Variable, Parameter, Shock, Equation, EXP
from .validation import validate_si_model, validate_model_consistency
from .logging_config import get_logger

from .parsing_tools import (from_dict_to_mat,
                            construct_equation_list,
                            find_max_lead_lag)

from .StateSpaceModel import StateSpaceModel

from .linalg.qz_solve import qz_solve
from scipy import sparse
from scipy.sparse.linalg import spsolve

import pandas as p

# Get module logger
logger = get_logger("dsge.si")

class LinLagExModel(StateSpaceModel):

    def __init__(self, yy, A, B, C, F, G, N, Q,
                 Aj, Bj, Cj, Fj, Gj,
                 Ainf, Binf, Cinf, Finf, Ginf,
                 t0=0,
                 shock_names=None, state_names=None, obs_names=None, parameter_names=None):

        self.A = A
        self.B = B
        self.C = C
        self.F = F
        self.G = G

        self.N = N
        self.Q = Q


        self.Aj = Aj
        self.Bj = Bj
        self.Cj = Cj
        self.Fj = Fj
        self.Gj = Gj

        self.Ainf = Ainf
        self.Binf = Binf
        self.Cinf = Cinf
        self.Finf = Finf
        self.Ginf = Ginf

        self.t0 = t0

        self.yy = yy

        self.shock_names = shock_names
        self.state_names = state_names
        self.orig_state_names = state_names.copy()
        self.obs_names = obs_names
        self.parameter_names = parameter_names 


    def find_max_it(self, p0, convergence_criteria=1e-10, max_it=5000, solver='python'):

        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)

        F = np.array(self.F(p0), dtype=float)

        if solver=='fortran':
            find_max_it = meyer_gohde_interface.mg.find_max_it
            j = find_max_it(Aj, Bj, Cj, Fj, Gj, F.shape[0], F.shape[1])
            return j
        else:
            for j in range(1, max_it):
                Amax = np.max(abs(Aj(j)))
                Bmax = np.max(abs(Bj(j)))
                Cmax = np.max(abs(Cj(j)))
                Fmax = np.max(abs(Fj(j)))
                Gmax = np.max(abs(Gj(j)))
                all_max = max(Amax, Bmax, Cmax, Fmax, Gmax)
                if all_max < convergence_criteria:
                    return j
            return j

    def companion_form(self,p0, solver='python'):

        if solver=='fortran':
            solve_ma = meyer_gohde_interface.mg.solve_ma_alt
        else:
            solve_ma = self.solve_ma_alt

        A = np.array(self.A(p0), dtype=float)
        B = np.array(self.B(p0), dtype=float)
        C = np.array(self.C(p0), dtype=float)
        F = np.array(self.F(p0), dtype=float)
        G = np.array(self.G(p0), dtype=float)
        N = np.array(self.N(p0), dtype=float)
        Q = np.array(self.Q(p0), dtype=float)

        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)

        Ainf = np.array(self.Ainf(p0), dtype=float)
        Binf = np.array(self.Binf(p0), dtype=float)
        Cinf = np.array(self.Cinf(p0), dtype=float)
        Ginf = np.array(self.Ginf(p0), dtype=float)
        Finf = np.array(self.Finf(p0), dtype=float)

        Imax = self.find_max_it(p0, solver=solver)
        MA_VECTOR, ALPHA, BETA, RC = solve_ma(A, B, C, F, G, N,
                                              Aj, Bj, Cj, Fj, Gj,
                                              Ainf, Binf, Cinf, Finf, Ginf, Imax-1)
        if len(MA_VECTOR.shape)==1: MA_VECTOR = MA_VECTOR[:,np.newaxis]
        
        nshocks = MA_VECTOR.shape[1]
        nvars = np.int(MA_VECTOR.shape[0]/Imax)
        TT = np.zeros([2*nvars,2*nvars])
        RR = np.zeros([2*nvars,nshocks*(Imax+1)])
        ma_coeffs = np.reshape(MA_VECTOR,([nvars,nshocks*Imax]),order='F')
        ma_coeffs2 = np.zeros([nvars,nshocks*Imax])
        for j in np.arange(nshocks):
            for i in np.arange(Imax):
                ma_coeffs2[:,j+i*nshocks] = ma_coeffs[:,j*Imax+i]
        RR[:nvars,:nshocks*Imax] = ma_coeffs2
        RR[:nvars,nshocks*Imax:] = ALPHA @ ma_coeffs2[:,nshocks*(Imax-1):]
        RR[nvars:,nshocks*Imax:] = RR[:nvars,nshocks*Imax:]
        TT[:nvars,nvars:] = ALPHA
        TT[nvars:,nvars:] = ALPHA
        return(TT,RR,Imax)

    def impulse_response(self, p0, h=20, solver='python'):

        if solver=='fortran':
            solve_ma = meyer_gohde_interface.mg.solve_ma_alt
        else:
            solve_ma = self.solve_ma_alt

        A = np.array(self.A(p0), dtype=float)
        B = np.array(self.B(p0), dtype=float)
        C = np.array(self.C(p0), dtype=float)
        F = np.array(self.F(p0), dtype=float)
        G = np.array(self.G(p0), dtype=float)
        N = np.array(self.N(p0), dtype=float)
        Q = np.array(self.Q(p0), dtype=float)

        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)

        Ainf = np.array(self.Ainf(p0), dtype=float)
        Binf = np.array(self.Binf(p0), dtype=float)
        Cinf = np.array(self.Cinf(p0), dtype=float)
        Ginf = np.array(self.Ginf(p0), dtype=float)
        Finf = np.array(self.Finf(p0),dtype=float)

        #ALPHA, BETA, RC = qz_solve(Ainf, Binf, Cinf, Finf, Ginf, N, 5,1)
        MA_VECTOR, ALPHA, BETA, RC = solve_ma(A, B, C, F, G, N,
                                              Aj, Bj, Cj, Fj, Gj,
                                              Ainf, Binf, Cinf, Finf, Ginf, h-1)
        if len(MA_VECTOR.shape)==1: MA_VECTOR = MA_VECTOR[:,np.newaxis]

        nshocks = MA_VECTOR.shape[1]
        nvars = MA_VECTOR.shape[0]/h
        irfs = {}
        i = 0

        for respi in MA_VECTOR.T:
            irfs[self.shock_names[i]] = p.DataFrame(np.reshape(respi, (h, int(nvars)))*np.sqrt(Q[i, i]), columns=self.state_names)
            i = i + 1
        return irfs

    def solve_ma_alt(self, A, B, C, F, G, N, Aj, Bj, Cj, Fj, Gj, Ainf, Binf, Cinf, Finf, Ginf, max_it):
        neq = A.shape[0]
        neps = F.shape[1]

        # Initialize RHS and LHS
        RHS = np.zeros(((max_it + 1) * neq, neps))
        LHS_data = []
        LHS_rows = []
        LHS_cols = []

        # Fill LHS data, rows, and columns
        # Replicating logic from Fortran
        LHS_data += list(B.T.ravel())
        LHS_rows += neq*[n for n in range(neq)]
        LHS_cols += [n for n in range(neq) for _ in range(neq)]

        LHS_data += list(A.T.ravel())
        LHS_rows += neq*[n for n in range(neq)]
        LHS_cols += [neq+n for n in range(neq) for _ in range(neq)]

        RHS[:neq,:] = -G - F @ N

        # Main loop to construct RHS and LHS matrices
        for i in range(1, max_it):
            # Get matrices for the current iteration
            Ajj = Aj(i)
            Bjj = Bj(i)
            Cjj = Cj(i)
            Fjj = Fj(i)
            Gjj = Gj(i)

            # Update A, B, C for each iteration
            A += Ajj 
            B += Bjj 
            C += Cjj 

            LHS_data += C.T.flatten().tolist()
            LHS_rows += neq*[n+i*neq for n in range(neq)]
            LHS_cols += [(i-1)*neq+n for n in range(neq) for _ in range(neq)]

            LHS_data += B.T.flatten().tolist()
            LHS_rows += neq*[n+i*neq for n in range(neq)]
            LHS_cols += [(i)*neq+n for n in range(neq) for _ in range(neq)]

            LHS_data += A.T.flatten().tolist()
            LHS_rows += neq*[n+i*neq for n in range(neq)]
            LHS_cols += [(i+1)*neq+n for n in range(neq) for _ in range(neq)]

            RHS_block =  -G @ np.linalg.matrix_power(N, i)  - F @ np.linalg.matrix_power(N, i+1)
            RHS[i * neq:(i + 1) * neq, :] = RHS_block


        ALPHA, BETA, RC = qz_solve(Ainf, Binf, Cinf, Finf, Ginf, N, neq, neps)
        RHS[max_it*neq:(1+max_it)*neq,:] = BETA @ np.linalg.matrix_power(N, max_it)
        
        LHS_data += list(-ALPHA.T.ravel())
        LHS_rows += neq*[n+(max_it)*neq for n in range(neq)]
        LHS_cols += [(max_it-1)*neq+n for n in range(neq) for _ in range(neq)]

        LHS_data += list(np.eye(neq).ravel())
        LHS_rows += neq*[n+(max_it)*neq for n in range(neq)]
        LHS_cols += [(max_it)*neq+n for n in range(neq) for _ in range(neq)]

        # Create sparse LHS matrix
        LHS = sparse.coo_matrix((LHS_data, (LHS_rows, LHS_cols)), shape=((max_it+1) * neq, (max_it+1) * neq)).tocsr()

        # Solve the linear system
        MA_VECTOR = spsolve(LHS, RHS)

        return MA_VECTOR, ALPHA, BETA, RC

    def system_matrices(self, p0, solver='python'):
        A = np.array(self.A(p0), dtype=float)
        B = np.array(self.B(p0), dtype=float)
        C = np.array(self.C(p0), dtype=float)
        F = np.array(self.F(p0), dtype=float)
        G = np.array(self.G(p0), dtype=float)
        N = np.array(self.N(p0), dtype=float)
        Q = np.array(self.Q(p0), dtype=float)
         
        Aj = lambda j: np.array(self.Aj(p0, j), dtype=float)
        Bj = lambda j: np.array(self.Bj(p0, j), dtype=float)
        Cj = lambda j: np.array(self.Cj(p0, j), dtype=float)
        Fj = lambda j: np.array(self.Fj(p0, j), dtype=float)
        Gj = lambda j: np.array(self.Gj(p0, j), dtype=float)
         
        Ainf = np.array(self.Ainf(p0), dtype=float)
        Binf = np.array(self.Binf(p0), dtype=float)
        Cinf = np.array(self.Cinf(p0), dtype=float)
        Ginf = np.array(self.Ginf(p0), dtype=float)
        Finf = np.array(self.Finf(p0),dtype=float)
         
        h = self.find_max_it(p0)
        if solver=='fortran':
            import meyer_gohde_interface
            ma_solve = meyer_gohde_interface.mg.solve_ma_alt
        else:
            ma_solve = self.solve_ma_alt
        MA_VECTOR, ALPHA, BETA, RC = ma_solve(A, B, C, F, G, N,
                                              Aj, Bj, Cj, Fj, Gj,
                                              Ainf, Binf, Cinf, Finf, Ginf, h-1)
        nshocks = MA_VECTOR.shape[1]
        nvars = MA_VECTOR.shape[0]/h
         
         
        ns = len(self.orig_state_names)
        TT = np.zeros((ns+(h-1)*nshocks,ns+(h-1)*nshocks))
        RR = np.zeros((ns+(h-1)*nshocks,nshocks))
         
        for i in range(h-1):
            TT[:ns,ns+i*nshocks:ns+(i+1)*nshocks] = MA_VECTOR[(i+1)*ns:(i+2)*ns,:]
         
        for i in range(h-2):
            TT[ns+(i+1)*nshocks:ns+(i+2)*nshocks, ns+i*nshocks:ns+(i+1)*nshocks] = np.eye(nshocks)
         
        RR[:ns,:] = MA_VECTOR[:ns,:]
        RR[ns:ns+nshocks] = np.eye(nshocks)

        CC = np.zeros((TT.shape[0]))

        QQ = self.Q(p0)
        DD = self.D(p0)
   
        ZZ = np.zeros((len(self.obs_names),TT.shape[0]))
        for i in range(len(self.obs_names)):
            ZZ[i,self.obs_ind[i]-1] = 1

        HH = np.zeros((len(self.obs_names), len(self.obs_names)))
        
        return CC,TT,RR,QQ,DD,ZZ,HH

from dsge.symbols import Variable, Equation, Shock, Parameter
from sympy.matrices import Matrix, zeros
import re

class SIDSGE(Base):


    def __init__(self, *kargs, **kwargs):
        super(SIDSGE, self).__init__(self, *kargs, **kwargs)


    @property
    def name(self):
        return self['name']

    @property
    def p(self):
        return self['p']

    @property
    def equations(self):
        return self['equations']

    @property
    def variables(self):
        return self.endo_variables + self.exo_variables

    @property
    def endo_variables(self):
        return self['var_ordering']

    @property
    def exo_variables(self):
        return self['exo_ordering']
        
    @property
    def shocks(self):
        return self['shk_ordering']

    @property
    def nvars(self):
        return len(self.variables)

    @property
    def nendo_vars(self):
        return len(self.endo_variables)
    
    @property
    def nobs_vars(self):
        return len(self['observables'])

    @property
    def nexo_vars(self):
        return len(self.exo_variables)

    @property
    def neqs(self):
        return len(self.equations)
    
    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return list(map(lambda x: self["calibration"][str(x)], self.parameters))

    @property
    def index(self):
        return self['index'][0]

    def __repr__(self):
        s="A DSGE model with {0} variables/equations and {1}".format(self.nvars, self.npara)
        return s

    def construct_sys_mat(self):
        subs_dict = dict()

        subs_dict.update( {v:0 for v in self.variables})
        subs_dict.update( {v(-1):0 for v in self.variables})
        subs_dict.update( {v(1):0 for v in self.variables})

        def d_rulej(i, j, order, varsel='var_ordering'):
            e = self['sequations'][i]
            v = self[varsel][j]
            ee = e.lhs - e.rhs
            j = self.index
     
            get_order = re.search('E\[([a-zA-Z0-9 -]+)\]', ee.__str__())

            if get_order is not None:
                e_order = eval(get_order.groups(0)[0], {j.name:j})
                gap = e_order + j
                d = ee.diff(EXP(e_order)(v(order))).subs({j:j+gap})
                # if d is not sympy.S.Zero:
                #     print e_order, gap, d, v(order)
            else:
                d = 0
            return d


        #--------------------------------------------------------------------------
        # 0 Matrices
        #--------------------------------------------------------------------------
        A = zeros(self.nendo_vars, self.nendo_vars)
        B = zeros(self.nendo_vars, self.nendo_vars)
        C = zeros(self.nendo_vars, self.nendo_vars)
        F = zeros(self.nendo_vars, self.nexo_vars)
        G = zeros(self.nendo_vars, self.nexo_vars)
        N = zeros(self.nexo_vars, self.nexo_vars)

        eq_i = 0
        j = self.index
        print(self['var_ordering'])
        for eq in self['sequations']:
            print(f"\r Differentiating equation {eq_i:3d}: {eq}")
            fvar = filter(lambda x: (x.date > 0)*(Variable(x.name) in self['var_ordering']), eq.atoms(Variable))

            #print(f"fvar:  {list(fvar)}")
            print(f"  {eq.atoms(Variable)}")
            for v in fvar:
                v_j = self['var_ordering'].index(Variable(v.name))
                A[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            cvar = filter(lambda x: (x.exp_date==0)*(x.date == 0)*((Variable(x.name) in self['var_ordering'])), eq.atoms(Variable))
            #print(f"  {list(cvar)}")
            print(f"  {eq.atoms(Variable)}")
            for v in cvar:
                v_j = self['var_ordering'].index(Variable(v.name))
                B[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict)# + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

                
            lvar = filter(lambda x: (x.exp_date==0)*(x.date < 0)*(Variable(x.name) in self['var_ordering']), eq.atoms(Variable))
            for v in lvar:
                v_j = self['var_ordering'].index(Variable(v.name))
                C[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            fshock = filter(lambda x: (x.date > 0)*(Variable(x.name) in self['exo_ordering']), eq.atoms(Variable))
            for v in fshock:
                v_j = self['exo_ordering'].index(Variable(v.name))
                F[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            
            cshock = filter(lambda x: (x.date == 0)*(Variable(x.name) in self['exo_ordering']), eq.atoms(Variable))

            for v in cshock:

                v_j = self['exo_ordering'].index(Variable(v.name)) 
                G[eq_i, v_j] = eq.set_eq_zero.diff(v).subs(subs_dict) + eq.set_eq_zero.diff(EXP(-j)(v)).subs(j, 0)

            eq_i += 1
            
        
        N = -1*N

        self.A = A
        self.B = B
        self.C = C
        self.F = F
        self.G = G
        self.N = N
        self.Q = self['QQ']

        self._A = self.lambdify(A) 
        self._B = self.lambdify(B)
        self._C = self.lambdify(C)
        self._F = self.lambdify(F)
        self._G = self.lambdify(G)
        self._N = self.lambdify(N)
        self._Q = self.lambdify(self['QQ'])
        


        def find_constant_obs(i):
            ee = self['obs_equations'][(self['observables'][i]).name]
            subsdict = dict(zip(self.variables, np.zeros(self.nvars)))
            return ee.subs(subsdict)

        #self.DD = Matrix(self.nobs_vars, 1, lambda i,j : find_constant_obs(i))
        self.DD = Matrix([find_constant_obs(i) for i in range(self.nobs_vars)])

        def find_obs_ind(i):
            ee = self['obs_equations'][(self['observables'][i]).name]
            v_ind = ee.atoms(Variable)

            assert len(v_ind) == 1
                
            return self['var_ordering'].index(list(v_ind)[0])+1

        self.obs_ind = [find_obs_ind(i) for i in range(self.nobs_vars)]
        #print(self.obs_ind)
        def d_ruleobs(i, j, order=0):
            ee = self['obs_equations'][(self['observables'][i]).name]
            v = self['var_ordering'][j]
            j = self.index
            d = ee.diff(v(order))
            return d

        self.Q1 = Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=0))
        self.Q2 = -1*Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=-1))

        self.Qlead = []
        self.Qleadempty = np.zeros((100, 1))
        for indi in np.arange(1, 100):
            self.Qlead.append(Matrix(self.nobs_vars, self.nendo_vars, lambda i, jj: d_ruleobs(i, jj, order=indi)))
            #print not(self.Qlead[indi-1] == sympy.zeros(self.nobs_vars, self.nendo_vars)), indi
            if not(self.Qlead[indi-1] == sympy.zeros(self.nobs_vars, self.nendo_vars)):
                self.Qleadempty[indi-1, :] = indi

        self.max_lead_observable = np.max(self.Qleadempty)

        
        print("zero matrices done")
        Aj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=1))     
        Bj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=0))     
        Cj= Matrix(self.nendo_vars, self.nendo_vars, lambda i,jj: d_rulej(i, jj, order=-1))    
        Fj = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rulej(i, jj, order=1, varsel='exo_ordering'))
        Gj = Matrix(self.nendo_vars, self.nexo_vars, lambda i,jj: d_rulej(i, jj, order=0, varsel='exo_ordering'))

        print("jth matrices done")

        self.Aj = Aj
        self.Bj = Bj
        self.Cj = Cj
        self.Fj = Fj
        self.Gj = Gj

        # HACK
        self['parameters'].append(self.index)
        self._Aj = self.lambdify(self.Aj) 
        self._Bj = self.lambdify(self.Bj) 
        self._Cj = self.lambdify(self.Cj) 
        self._Fj = self.lambdify(self.Fj) 
        self._Gj = self.lambdify(self.Gj)
        self['parameters'].pop()
        Ainf= A.copy()
        Binf= B.copy()
        Cinf= C.copy()
        Finf= F.copy()
        Ginf= G.copy()

        for ii in np.arange(0, self.nendo_vars):                                                                                         
            for jj in np.arange(0, self.nendo_vars):                                                                                     
                ad = sympy.Sum(Aj[ii, jj], (self.index, 1, sympy.oo)).doit()
                bd = sympy.Sum(Bj[ii, jj], (self.index, 1, sympy.oo)).doit()
                cd = sympy.Sum(Cj[ii, jj], (self.index, 1, sympy.oo)).doit()
                
                context = [(s.name,s) for s in
                           self['var_ordering'] + self['parameters'] + self['index'] +
                           self['shk_ordering'] + self['exo_ordering'] + self['other_para']]
              
                context = dict(context)                                                             

                context['inf'] = sympy.oo
                context['Piecewise'] = lambda x, *args: x[0]      

                for f in [sympy.log, sympy.exp,
                          sympy.sin, sympy.cos, sympy.tan,
                          sympy.asin, sympy.acos, sympy.atan,
                          sympy.sinh, sympy.cosh, sympy.tanh,
                          sympy.pi, sympy.sign]:
                    context[str(f)] = f                                                                                    

                context['Sum'] = sympy.Sum
                context['oo'] = sympy.oo
                context['Abs'] = sympy.Abs

                                              
                ad = eval(ad.__str__(), context)
                bd = eval(bd.__str__(), context)
                
                cd = eval(cd.__str__(), context)
                Ainf[ii, jj] = Ainf[ii, jj] + ad
                Binf[ii, jj] = Binf[ii, jj] + bd
                Cinf[ii, jj] = Cinf[ii, jj] + cd

        self.Ainf = Ainf
        self.Binf = Binf
        self.Cinf = Cinf
        self.Ginf = Ginf
        self.Finf = Finf

        
        self._Ainf = self.lambdify(self.Ainf) 
        self._Binf = self.lambdify(self.Binf) 
        self._Cinf = self.lambdify(self.Cinf) 
        self._Finf = self.lambdify(self.Finf) 
        self._Ginf = self.lambdify(self.Ginf) 

        self._D = self.lambdify(self.DD)
                
        print("System matrices constructed.")

    def compile_model(self):

        self.construct_sys_mat()
        A = lambda x: self._A(x)
        B = lambda x: self._B(x)
        C = lambda x: self._C(x)
        F = lambda x: self._F(x)
        G = lambda x: self._G(x)
        N = lambda x: self._N(x)
        Q = lambda x: self._Q(x)
        
        Aj = lambda x, j: self._Aj(np.append(x, j))
        Bj = lambda x, j: self._Bj(np.append(x, j))
        Cj = lambda x, j: self._Cj(np.append(x, j))
        Fj = lambda x, j: self._Fj(np.append(x, j))
        Gj = lambda x, j: self._Gj(np.append(x, j))
        
        Ainf = lambda x: self._Ainf(x)
        Binf = lambda x: self._Binf(x)
        Cinf = lambda x: self._Cinf(x)
        Finf = lambda x: self._Finf(x)
        Ginf = lambda x: self._Ginf(x)

        from dsge.data import read_data_file
        print(self['observables'])
        if "observables" not in self:
            self["observables"] = self["variables"].copy()
            self["obs_equations"] = dict(self["observables"], self["observables"])

        if "data" in self["__data__"]["estimation"]:
            data = read_data_file(
                self["__data__"]["estimation"]["data"], self["observables"]
            )
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        

        mod = LinLagExModel(data, A, B, C, F, G, N, Q, 
                            Aj, Bj, Cj, Fj, Gj,
                            Ainf, Binf, Cinf, Finf, Ginf,
                            t0=0,
                            shock_names=list(map(lambda x: str(x), self.shocks)),
                            state_names=list(map(lambda x: str(x), self.endo_variables)), 
                            obs_names=list(map(str, self['observables'])),
                            parameter_names=list(map(str,self.parameters)))

        mod.D = self._D
        mod.obs_ind = self.obs_ind
        return mod
    def __print_matrix_list(self, mat_list, wrapper_function):
        pstr = ''
        for key in mat_list:
            for i in np.arange(mat_list[key].rows):
                for j in np.arange(mat_list[key].cols):
                    if mat_list[key][i, j] != 0:
                        fstr = wrapper_function(mat_list[key][i, j])
                        pstr += "{0}({1:2d}, {2:2d}) = {3};\n".format(key, i+1, j+1, fstr)
        return pstr
        


    def create_fortran_model(self, model_dir='_fortress_tmp/', **kwargs):
        fortran_template_file = 'sticky_information_fortran_template.f90'
        self.construct_sys_mat()
        model = self.compile_model()

        with open(fortran_template_file) as f:
            template = f.read()

        npara = len(self['parameters'])

        para = sympy.IndexedBase("para", shape=(npara + 1,))

        from sympy.printing import fcode
        fortran_subs = dict(
            zip(
                [sympy.symbols("garbage")] + self['parameters'],
                para,
            )
        )
        fortran_subs[0] = 0.0
        fortran_subs[1] = 1.0
        fortran_subs[100] = 100.0
        fortran_subs[2] = 2.0
        fortran_subs[400] = 400.0
        fortran_subs[4] = 4.0

        context_tuple = [(str(p), p) for p in self['parameters']] + [
            (p.name, p) for p in self["auxiliary_parameters"].keys()
        ]

        context = dict(context_tuple)
        context["exp"] = sympy.exp
        context["log"] = sympy.log

        to_replace = {}
        for p in self['auxiliary_parameters'].keys():
            to_replace[p] = eval(str(self["auxiliary_parameters"][p]), context)

        to_replace = list(to_replace.items())
        from itertools import permutations

        edges = [
            (i, j)
            for i, j in permutations(to_replace, 2)
            if type(i[1]) not in [float, int] and i[1].has(j[0])
        ]

        from sympy import default_sort_key, topological_sort

        para_func = topological_sort([to_replace, edges], default_sort_key)

        symbolic_matrices = {'A': self.A,
                             'B' : self.B,
                             'C' : self.C,
                             'F' : self.F,
                             'G' : self.G,
                             'N' : self.N,
                             'DD2': self.DD,
                             # 'HH': self.HH,
                             'self%QQ' : self.Q,
                             'Aj' : self.Aj,
                             'Bj' : self.Bj,
                             'Cj' : self.Cj,
                             'Fj' : self.Fj,
                             'Gj' : self.Gj,
                             'Ainf' : self.Ainf, 
                             'Binf' : self.Binf, 
                             'Cinf' : self.Cinf, 
                             'Finf' : self.Finf, 
                             'Ginf' : self.Ginf}
        
        j_matrices = [
            fcode(
                (mat.subs(para_func)).subs(fortran_subs),
                assign_to=n,
                source_format="free",
                standard=95,
                contract=False,
            )
            for n, mat in symbolic_matrices.items()
            if n in ['Aj','Bj','Cj','Fj','Gj']
        ]

        zero_matrices = [
            fcode(
                (mat.subs(para_func)).subs(fortran_subs),
                assign_to=n,
                source_format="free",
                standard=95,
                contract=False,
            )
            for n, mat in symbolic_matrices.items()
            if n in ['A','B','C','F','G','N','self%QQ','DD2']
        ] 

        inf_matrices = [
            fcode(
                (mat.subs(para_func)).subs(fortran_subs),
                assign_to=n,
                source_format="free",
                standard=95,
                contract=False,
            )
            for n, mat in symbolic_matrices.items()
            if n in ['Ainf','Binf','Cinf','Finf','Ginf']
        ]

        
        pmsv = ','.join([str(float(p0)) + '_wp' for p0 in self.p0()])


        smc_file =template.format(name='stick_information_model',
                                  npara=len(self.parameters), 
                                  neps=len(self.shocks), #self.name,
                                  nendo_vars=self.nendo_vars,
                                  yy=model.yy,
                                  model=model,
                                  obs_ind=self.obs_ind,
                                  pmsv=pmsv,
                                  zero_matrices='\n'.join(zero_matrices), 
                                  j_matrices='\n'.join(j_matrices), 
                                  inf_matrices='\n'.join(inf_matrices))

        
        with open('meyer_gohde_interface.f90') as f:
            mg = f.read()

        from fortress import make_smc

        paths = {'f90':'mpif90',
                 'lib_path':'/cmc/home/m1eph00/miniconda2/envs/proxy-svar/lib/',
                 'inc_path':'/cmc/home/m1eph00/miniconda2/envs/proxy-svar/include/'}

        makefile = """
LIB={lib_path}
INC={inc_path}
FPP=fypp
FRUIT?=-I$(INC)/fruit -L$(LIB) -lfruit -Wl,-rpath=$(LIB)
FLAP?=-I$(INC)/flap -L$(LIB) -lflap
FORTRESS?=-I$(INC)/fortress -L$(LIB)/fortress -lfortress
JSON?=-I$(INC)/json-fortran -L$(LIB)/json-fortran -ljsonfortran
MKLROOT?=/opt/intel/2019.1/mkl
MKL?=-I$(MKLROOT)/include/ -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed  -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
FC={f90} -O3 -ffree-line-length-1000 #-Wall -fcheck=all -g -fbacktrace
smc_driver : {model_file} smc_driver.f90 mg.f90
\t$(FC) $^  -I. -Wl,--start-group $(FORTRESS) $(JSON) $(FLAP) $(FRUIT) $(OPENBLAS) -Wl,--end-group $(MKL) -o smc 
check_likelihood : {model_file} check_likelihood.f90 mg.f90
\t$(FC) $^  -I. -Wl,--start-group $(FORTRESS) $(JSON) $(FLAP) $(FRUIT) -l{lapack} -Wl,--end-group $(MKL) -o check_likelihood 
""".format(model_file='model_t.f90', lapack='openblas', **paths)

        other_files = {'data.txt': model.yy,          
                       'prior.txt': 'prior.txt',
                       'mg.f90': mg,
                       'makefile': makefile}


        make_smc(smc_file, output_directory=model_dir, other_files=other_files, **paths, **kwargs)

        from dsge.Prior import construct_prior
        from dsge.translate import write_prior_file
        prior = None
        if "prior" in self["__data__"]["estimation"]:
            prior = construct_prior(
                self["__data__"]["estimation"]["prior"], self.parameters
            )

        from dsge.Prior import Prior as pri

        write_prior_file(pri(prior), model_dir)           


def read_si(model_yaml: Dict[str, Any]) -> SIDSGE:
    """
    Read a Sticky Information DSGE model from a YAML specification.
    
    Args:
        model_yaml: Dictionary containing the model specification
        
    Returns:
        A SIDSGE model instance
        
    Raises:
        ValueError: If the model specification contains errors
    """
    logger.info("Reading SI model from YAML specification")
    
    dec = model_yaml['declarations']
    cal = model_yaml['calibration']

    name = dec['name']
    logger.debug(f"Model name: {name}")

    var_ordering = [Variable(v) for v in dec['variables']]
    par_ordering = [Parameter(v) for v in dec['parameters']]
    shk_ordering = [Shock(v) for v in dec['shocks']]
    exo_ordering = [Variable('exo_'+v) for v in dec['shocks']]

    logger.debug(f"Model has {len(var_ordering)} variables, {len(shk_ordering)} shocks, and {len(par_ordering)} parameters")
    
    exo_subs_dict = dict(zip(shk_ordering, exo_ordering))

    if 'auxiliary_parameters' in dec:
        other_para = [Parameter(v) for v in dec['auxiliary_parameters']]
        logger.debug(f"Model has {len(other_para)} auxiliary parameters")
    else:
        other_para = []

    if 'observables' in dec:
        observables = [Variable(v) for v in dec['observables']]
        obs_equations = model_yaml['equations']['observables']
        logger.debug(f"Model has {len(observables)} observables")
    else:
        observables = [Variable(v) for v in dec["variables"]]
        obs_equations = {v: v for v in dec['variables']}
        logger.debug("Using variables as observables")

    index = [Parameter(j) for j in dec['index']]
    logger.debug(f"Index variable: {dec['index']}")
    
    context = {s.name:s for s in
               (var_ordering + par_ordering + index + shk_ordering + other_para)}

    context['EXP'] = EXP
    context['inf'] = sympy.oo
    context['SUM'] = sympy.Sum
    rcontext = context.copy()
    rcontext['SUM'] = lambda x, d: x

    for obs in obs_equations.items():
        obs_equations[obs[0]] = eval(obs[1], context)

    if "model" in model_yaml["equations"]:
        raw_equations = model_yaml["equations"]["model"]
    else:
        raw_equations = model_yaml["equations"]
    
    logger.debug(f"Processing {len(raw_equations)} model equations")
    
    sum_rem_equations = [eq.subs(exo_subs_dict)
                         for eq in construct_equation_list(raw_equations, rcontext)]
    equations = [eq.subs(exo_subs_dict)
                 for eq in construct_equation_list(raw_equations, context)]
                 
    # Validate SI model constraints
    logger.info("Validating SI model constraints")
    index_var = dec['index']
    validation_errors = validate_si_model({
        'equations': equations,
        'variables': var_ordering,
        'index': index_var
    }, index_var)
    
    if validation_errors:
        for error in validation_errors:
            logger.error(error)
        raise ValueError(
            "SI model validation failed. The following errors were found:\n" + 
            "\n".join(validation_errors)
        )
    
    # General model consistency checks (warning only)
    warnings = validate_model_consistency({
        'equations': equations, 
        'variables': var_ordering
    })
    for warning in warnings:
        logger.warning(warning)

    max_lead_endo, max_lag_endo = find_max_lead_lag(equations, var_ordering)
 

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

    for exo, shk in exo_subs_dict.items():
        exo_equations.append(Equation(exo, shk))

    print('Exo equations:', exo_equations, exo_subs_dict)
    calibration = model_yaml['calibration']['parameters']
    p = calibration


    if 'auxiliary_parameters' in model_yaml['calibration']:
        para_func = model_yaml['calibration']['auxiliary_parameters']
    else:
        para_func = []

    func_dict = dict()
        
    opr = other_para
    
    if 'covariance' in cal:
        QQ = from_dict_to_mat(cal['covariance'], shk_ordering, context)
    else:
        print('No covariance matrix provided. Assuming identity matrix.')
        QQ = sympy.eye(len(shk_ordering))

    if 'estimation' not in model_yaml:
        model_yaml['estimation'] = {}
    
    if "auxiliary_parameters" not in cal:
        cal["auxiliary_parameters"] = {}
    else:
        cal['auxiliary_parameters'] = {op:
                                       sympy.sympify(cal["auxiliary_parameters"][str(op)],
                                                  {str(p): p for p in
                                                   par_ordering+other_para})
                                       for op in other_para}



    model_dict = {
        'var_ordering': var_ordering, 
        'exo_ordering': exo_ordering, 
        'parameters': par_ordering, 
        'shk_ordering': shk_ordering, 
        'index': index, 
        'equations': equations, 
        'exo_equations': exo_equations, 
        'sequations': sum_rem_equations, 
        'calibration': calibration, 
        'p': calibration, 
        'QQ': QQ, 
        'other_para': other_para,
        'auxiliary_parameters': cal['auxiliary_parameters'],
        'observables': observables, 
        'obs_equations': obs_equations, 
        'name': name, 
        '__data__': model_yaml, 
        'para_func_d': func_dict
    }

    logger.info(f"SI model {name} creation complete with {len(var_ordering)} variables")
    model = SIDSGE(**model_dict)
    return model




