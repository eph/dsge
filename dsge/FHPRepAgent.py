#!/usr/bin/env python3
import numpy as np
import sympy



from sympy import sympify
from sympy.utilities.lambdify import lambdify

from .symbols import (Variable,
                      Shock,
                      Parameter)


from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel
from .parse_yaml import from_dict_to_mat, construct_equation_list, read_yaml


class LinearDSGEforFHPRepAgent(LinearDSGEModel):
    def __init__(
            self,
            yy,
            alpha0_cycle,
            alpha1_cycle,
            beta0_cycle,
            alphaC_cycle, alphaF_cycle, alphaB_cycle, betaS_cycle,
            alpha0_trend, alpha1_trend, betaV_trend,
            alphaC_trend, alphaF_trend, alphaB_trend,
            value_gammaC, value_gamma, value_Cx, value_Cs, P, R, QQ,
        DD,
        ZZ,
        HH,
        k,
        t0=0,
        shock_names=None,
        state_names=None,
        obs_names=None,
        prior=None,
        parameter_names=None
    ):

        if len(yy.shape) < 2:
            yy = np.swapaxes(np.atleast_2d(yy), 0, 1)

        self.yy = yy

        self.alpha0_cycle = alpha0_cycle
        self.alpha1_cycle = alpha1_cycle
        self.beta0_cycle = beta0_cycle
        self.alphaC_cycle = alphaC_cycle
        self.alphaF_cycle = alphaF_cycle
        self.alphaB_cycle = alphaB_cycle
        self.betaS_cycle = betaS_cycle
        self.alpha0_trend = alpha0_trend
        self.alpha1_trend = alpha1_trend
        self.betaV_trend = betaV_trend
        self.alphaC_trend = alphaC_trend
        self.alphaF_trend = alphaF_trend
        self.alphaB_trend = alphaB_trend
        self.value_gammaC = value_gammaC
        self.value_gamma = value_gamma
        self.value_Cx = value_Cx
        self.value_Cs = value_Cs

        self.P = P
        self.R = R
        self.QQ = QQ
        self.DD = DD
        self.ZZ = ZZ
        self.HH = HH

        self.t0 = t0

        self.shock_names = shock_names
        self.state_names = state_names
        self.obs_names = obs_names
        self.parameter_names = parameter_names

        self.prior = prior
        self.k = k

    def system_matrices(self, p0):

       alpha0_cycle  = self.alpha0_cycle(p0)
       alpha1_cycle  = self.alpha1_cycle(p0)
       beta0_cycle   = self.beta0_cycle(p0)
       alphaC_cycle  = self.alphaC_cycle(p0)
       alphaF_cycle  = self.alphaF_cycle(p0)
       alphaB_cycle  = self.alphaB_cycle(p0)
       betaS_cycle   = self.betaS_cycle(p0)
       alpha0_trend  = self.alpha0_trend(p0)
       alpha1_trend  = self.alpha1_trend(p0)
       betaV_trend   = self.betaV_trend(p0)
       alphaC_trend  = self.alphaC_trend(p0)
       alphaF_trend  = self.alphaF_trend(p0)
       alphaB_trend  = self.alphaB_trend(p0)
       value_gammaC  = self.value_gammaC(p0)
       value_gamma   = self.value_gamma(p0)
       value_Cx      = self.value_Cx(p0)
       value_Cs      = self.value_Cs(p0)

       P = self.P(p0)
       R = self.R(p0)
       A_cycle = np.linalg.inv(alpha0_cycle) @ alpha1_cycle
       B_cycle = np.linalg.inv(alpha0_cycle) @ beta0_cycle
       A_trend = np.linalg.inv(alpha0_trend) @ alpha1_trend
       B_trend = np.linalg.inv(alpha0_trend) @ betaV_trend

       for k in range(1, self.k+1):
           A_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ alphaB_cycle
           B_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ (alphaF_cycle @ B_cycle @ P + betaS_cycle)

           A_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ alphaB_trend
           B_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ (alphaF_trend @ B_trend)

           A_cycle = A_cycle_new
           B_cycle = B_cycle_new

           A_trend = A_trend_new
           B_trend = B_trend_new

       nx = A_cycle.shape[0]
       zero = np.zeros((nx, nx))
       zeroV = np.zeros_like(B_trend)
       nx,ns = B_cycle.shape
       zeroS = np.zeros((nx, ns))


       TT = np.r_[
           np.c_[B_trend @ value_gamma @ value_Cx, A_cycle, A_trend, B_trend @ value_gammaC, B_cycle @ P + B_trend @ value_gamma @ value_Cs],
           np.c_[zero                            , A_cycle, zero   , zeroV                 , B_cycle @ P                                   ],
           np.c_[B_trend @ value_gamma @ value_Cx, zero   , A_trend, B_trend @ value_gammaC, B_trend @ value_gamma @ value_Cs              ],
           np.c_[value_gamma @ value_Cx          , zeroV.T, zeroV.T, value_gammaC          , value_gamma @ value_Cs                        ],
           np.c_[zeroS.T                         , zeroS.T, zeroS.T, zeroS.T@zeroV         , P                                             ]
       ]

       RR = np.r_[B_cycle @ R,
                  B_cycle @ R,
                  zeroS,
                  zeroV.T @ zeroS,
                  R]

       CC = np.zeros((A_cycle.shape[0], 1))
       QQ = self.QQ(p0)
       DD = self.DD(p0)
       ZZ = self.ZZ(p0)
       HH = self.HH(p0)


       return CC, TT, RR, QQ, DD, ZZ, HH


class FHPRepAgent(dict):

    def __init___():
        pass

    def p0(self):
        return list(map(lambda x: self["calibration"]['parameters'][str(x)], self['parameters']))


    @classmethod
    def read(cls, model_file, k=None):
        model_yaml = read_yaml(model_file)

        dec = model_yaml['declarations']
        variables = [Variable(v) for v in dec['variables']]
        values = [Variable(v) for v in dec['values']]
        value_updates = [Variable(v) for v in dec['value_updates']]
        shocks = [Variable(v) for v in dec['shocks']]
        innovations = [Shock(v) for v in dec['innovations']]
        parameters = [Parameter(v) for v in dec['parameters']]

        if "para_func" in dec:
            other_para = [Parameter(v) for v in dec["para_func"]]

            other_para = {op: sympify(model_yaml['calibration']["para_func"][op.name],
                                      {str(x): x for x in parameters+other_para})
                          for op in other_para}

        else:
            other_para = {}

        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
        else:
            measurement_errors = None

        context = {s.name: s
                   for s in (variables +
                          values + value_updates +
                          shocks + innovations +
                          parameters + list(other_para.keys()))}

        if "observables" in dec:
            observables = [Variable(v)  for v in dec["observables"]]
            obs_equations = {o: sympify(model_yaml["equations"]["observables"][o], context)
                             for o in observables}
        else:
            observables = [Variable(v) for v in dec["variables"]]
            obs_equations = {v: v for v in observables}

        # set up the equations
        yaml_eq = model_yaml['model']
        equations = {}
        if 'static' in yaml_eq:
            equations['static'] = construct_equation_list(yaml_eq['static'], context)
        else:
            equations['static'] = []

        equations['cycle'] = {'terminal': construct_equation_list(yaml_eq['cycle']['terminal'], context),
                              'plan': construct_equation_list(yaml_eq['cycle']['plan'], context)}

        assert len(equations['cycle']['terminal']) + len(equations['static']) == len(variables)
        assert len(equations['cycle']['plan']) + len(equations['static']) == len(variables)

        equations['trend'] = {'terminal': construct_equation_list(yaml_eq['trend']['terminal'], context),
                              'plan': construct_equation_list(yaml_eq['trend']['plan'], context)}

        assert len(equations['trend']['terminal']) + len(equations['static']) == len(variables)
        assert len(equations['trend']['plan']) + len(equations['static']) == len(variables)

        equations['value'] = {}
        equations['value']['function'] = construct_equation_list(yaml_eq['value']['function'], context)
        equations['value']['update'] = construct_equation_list(yaml_eq['value']['update'], context)

        equations['shocks'] = construct_equation_list(yaml_eq['shocks'], context)

        cov = model_yaml['calibration']["covariance"]
        QQ = from_dict_to_mat(cov, innovations, context)

        me_dict = {}
        if 'measurement_errors' in model_yaml['calibration']:
            me_dict = model_yaml['calibration']['measurement_errors']

        if measurement_errors is not None:
            HH = from_dict_to_mat(me_dict, measurement_errors, context)
        else:
            HH = from_dict_to_mat(me_dict, observables, context)


        model_dict = {'variables': variables,
                      'values': values,
                      'value_updates': value_updates,
                      'shocks': shocks,
                      'innovations': innovations,
                      'parameters': parameters,
                      'other_para': other_para,
                      'context': context,
                      'equations': equations,
                      'calibration': model_yaml['calibration'],
                      'estimation': model_yaml['estimation'] if 'estimation' in model_yaml else {},
                      'observables': observables,
                      'obs_equations': obs_equations,
                      'QQ': QQ,
                      'HH': HH,
                      'k': dec['k']}


        return cls(**model_dict)

    def compile_model(self, k=None):

        k = self['k'] if k is None else k

        nv = len(self['variables'])
        ns = len(self['shocks'])
        nval = len(self['values'])
        v = self['variables']
        cycle_equation = self['equations']['cycle']['terminal'] + self['equations']['static']
        self.alpha0_cycle = sympy.Matrix(nv, nv , lambda i, j: cycle_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_cycle = sympy.Matrix(nv, nv , lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.beta0_cycle = sympy.Matrix(nv, ns , lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['shocks'][j]))

        cycle_equation = self['equations']['cycle']['plan'] + self['equations']['static']
        self.alphaC_cycle = sympy.Matrix(nv, nv, lambda i, j: cycle_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaS_cycle = sympy.Matrix(nv, ns, lambda i ,j: -cycle_equation[i].set_eq_zero.diff(self['shocks'][j]))
        trend_equation = self['equations']['trend']['terminal'] + self['equations']['static']
        self.alpha0_trend = sympy.Matrix(nv, nv , lambda i, j: trend_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_trend = sympy.Matrix(nv, nv , lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaV_trend = sympy.Matrix(nv, nval , lambda i, j: -trend_equation[i].set_eq_zero.diff(self['values'][j]))

        trend_equation = self['equations']['trend']['plan'] + self['equations']['static']
        self.alphaC_trend = sympy.Matrix(nv, nv, lambda i, j: trend_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_equation[i].set_eq_zero.diff(self['variables'][j](-1)))

        value_equation = self['equations']['value']['function']
        lhs = sympy.Matrix(nval, nval, lambda i, j: value_equation[i].set_eq_zero.diff(self['values'][j]))
        self.value_gammaC = lhs.inv() * sympy.Matrix(nval, nval, lambda i, j: -value_equation[i].set_eq_zero.diff(self['values'][j](-1)))
        self.value_gamma = lhs.inv() * sympy.Matrix(nval, nval, lambda i, j: -value_equation[i].set_eq_zero.diff(self['value_updates'][j](-1)))

        value_equation = self['equations']['value']['update']
        lhs = sympy.Matrix(nval, nval, lambda i, j: value_equation[i].set_eq_zero.diff(self['value_updates'][j]))
        self.value_Cx = lhs.inv() * sympy.Matrix(nval, nv, lambda i, j: -value_equation[i].set_eq_zero.diff(self['variables'][j]))
        self.value_Cs = lhs.inv() * sympy.Matrix(nval, ns, lambda i, j: -value_equation[i].set_eq_zero.diff(self['shocks'][j]))

        exo_equation = self['equations']['shocks']
        lhs = sympy.Matrix(ns, ns, lambda i, j: exo_equation[i].set_eq_zero.diff(self['shocks'][j]))
        self.P = lhs.inv() * sympy.Matrix(ns, ns, lambda i, j: -exo_equation[i].set_eq_zero.diff(self['shocks'][j](-1)))
        self.R = lhs.inv() * sympy.Matrix(ns, ns, lambda i, j: -exo_equation[i].set_eq_zero.diff(self['innovations'][j]))

        system_matrices = [self.alpha0_cycle, self.alpha1_cycle, self.beta0_cycle,
                           self.alphaC_cycle, self.alphaF_cycle, self.alphaB_cycle, self.betaS_cycle,
                           self.alpha0_trend, self.alpha1_trend, self.betaV_trend,
                           self.alphaC_trend, self.alphaF_trend, self.alphaB_trend,
                           self.value_gammaC, self.value_gamma, self.value_Cx, self.value_Cs,
                           self.P, self.R]


        all_para = self['parameters'] + list(self['other_para'].keys())

        lambdify_system_matrices = [lambdify(all_para, s)
                                    for s in system_matrices]

        intermediate_parameters = lambdify(self['parameters'], list(self['other_para'].values()))
        def expand_intermediate_parameters(f):
            def function_with_expanded_parameters(para):
                return f(*para, *intermediate_parameters(*para))
            return function_with_expanded_parameters


        alpha0_cycle = expand_intermediate_parameters(lambdify_system_matrices[0])
        alpha1_cycle = expand_intermediate_parameters(lambdify_system_matrices[1])
        beta0_cycle = expand_intermediate_parameters(lambdify_system_matrices[2])
        alphaC_cycle = expand_intermediate_parameters(lambdify_system_matrices[3])
        alphaF_cycle = expand_intermediate_parameters(lambdify_system_matrices[4])
        alphaB_cycle = expand_intermediate_parameters(lambdify_system_matrices[5])
        betaS_cycle = expand_intermediate_parameters(lambdify_system_matrices[6])
        alpha0_trend = expand_intermediate_parameters(lambdify_system_matrices[7])
        alpha1_trend = expand_intermediate_parameters(lambdify_system_matrices[8])
        betaV_trend = expand_intermediate_parameters(lambdify_system_matrices[9])
        alphaC_trend = expand_intermediate_parameters(lambdify_system_matrices[10])
        alphaF_trend = expand_intermediate_parameters(lambdify_system_matrices[11])
        alphaB_trend = expand_intermediate_parameters(lambdify_system_matrices[12])
        value_gammaC = expand_intermediate_parameters(lambdify_system_matrices[13])
        value_gamma = expand_intermediate_parameters(lambdify_system_matrices[14])
        value_Cx = expand_intermediate_parameters(lambdify_system_matrices[15])
        value_Cs = expand_intermediate_parameters(lambdify_system_matrices[16])
        P = expand_intermediate_parameters(lambdify_system_matrices[17])
        R = expand_intermediate_parameters(lambdify_system_matrices[18])

        p0 = self.p0()

        if "data" in self["estimation"]:
            data = read_data_file(self["estimation"]["data"], self["observables"])
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        QQ = expand_intermediate_parameters(lambdify(all_para, self['QQ']))
        nobs = len(self['observables'])
        all_obj = self['variables']+self['shocks']+self['innovations']+self['values']+self['value_updates']
        subs_dict = {}
        subs_dict.update({v: 0 for v in all_obj})
        subs_dict.update({v(-1): 0 for v in all_obj})
        subs_dict.update({v(+1): 0 for v in all_obj})
        DD = sympy.Matrix(nobs, 1, lambda i,j: self['obs_equations'][self['observables'][i]]
                          .subs(subs_dict))
        DD = expand_intermediate_parameters(lambdify(all_para, DD))

        ZZ = sympy.Matrix(nobs, nv, lambda i, j: self['obs_equations'][self['observables'][i]].diff(self['variables'][j]))
        ZZ = expand_intermediate_parameters(lambdify(all_para, ZZ))

        HH = expand_intermediate_parameters(lambdify(all_para, self['HH']))


        prior = None
        if "prior" in self["estimation"]:
            prior = construct_prior(self["estimation"]["prior"], self.parameters)

        from .Prior import Prior as pri



        shock_names = [str(x) for x in self['innovations']]
        obs_names = [str(x) for x in self['observables']]
        state_names = ([str(x) for x in self['variables']]
                     + [str(x)+'_cycle' for x in self['variables']]
                     + [str(x)+'_trend' for x in self['variables']]
                     + [str(x) for x in self['values']]
                     + [str(x) for x in self['shocks']])

        parameter_names = [str(x) for x in self['parameters']]

        linmod = LinearDSGEforFHPRepAgent(data, alpha0_cycle, alpha1_cycle,
                                          beta0_cycle, alphaC_cycle, alphaF_cycle, alphaB_cycle, betaS_cycle,
                                          alpha0_trend, alpha1_trend, betaV_trend, alphaC_trend, alphaF_trend,
                                          alphaB_trend, value_gammaC, value_gamma, value_Cx, value_Cs, P, R, QQ, DD, ZZ,
                                          HH, k, t0=0,
                                          shock_names=shock_names,
                                          state_names=state_names,
                                          obs_names=obs_names,
                                          prior=None,
                                          parameter_names=parameter_names)
        return linmod
