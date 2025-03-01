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
from .parsing_tools import from_dict_to_mat, construct_equation_list

from sympy.printing import fcode
from sympy.printing.fortran import FCodePrinter

from .Base import Base

# Define a new function
def _print_Integer(self, expr):
    if expr == 0:
        return '0.0'
    else:
        return super(FCodePrinter, self)._print_Integer(expr)

# Monkey patching FCodePrinter's _print_Integer
FCodePrinter._print_Integer = _print_Integer



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
        expectations=0,
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

        self.expectations = expectations
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

       A_cycle_history = {}
       B_cycle_history = {}
       A_trend_history = {}
       B_trend_history = {}

       if self.expectations > 0:
           A_cycle_history[0] = A_cycle.copy()
           B_cycle_history[0] = B_cycle.copy()

           A_trend_history[0] = A_trend.copy()
           B_trend_history[0] = B_trend.copy()

       for k in range(1, self.k+1):
           A_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ alphaB_cycle
           B_cycle_new = np.linalg.inv(alphaC_cycle - alphaF_cycle @ A_cycle) @ (alphaF_cycle @ B_cycle @ P + betaS_cycle)

           A_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ alphaB_trend
           B_trend_new = np.linalg.inv(alphaC_trend - alphaF_trend @ A_trend) @ (alphaF_trend @ B_trend)

           A_cycle = A_cycle_new
           B_cycle = B_cycle_new

           A_trend = A_trend_new
           B_trend = B_trend_new

           if self.expectations > 0:
               A_cycle_history[k] = A_cycle.copy()
               B_cycle_history[k] = B_cycle.copy()

               A_trend_history[k] = A_trend.copy()
               B_trend_history[k] = B_trend.copy()

           self.A_cycle = A_cycle
           self.B_cycle = B_cycle
           self.A_trend = A_trend
           self.B_trend = B_trend


       nx = A_cycle.shape[0]
       zero = np.zeros((nx, nx))
       zeroV = np.zeros_like(B_trend)
       nx,ns = B_cycle.shape
       zeroS = np.zeros((nx, ns))
       nv = B_trend.shape[1]

       TT = np.r_[
           np.c_[B_trend @ value_gamma @ value_Cx, A_cycle, A_trend, B_trend @ value_gammaC, B_cycle @ P + B_trend @ value_gamma @ value_Cs],
           np.c_[zero                            , A_cycle, zero   , zeroV                 , B_cycle @ P                                   ],
           np.c_[B_trend @ value_gamma @ value_Cx, zero   , A_trend, B_trend @ value_gammaC, B_trend @ value_gamma @ value_Cs              ],
           np.c_[value_gamma @ value_Cx          , zeroV.T, zeroV.T, value_gammaC          , value_gamma @ value_Cs                        ],
           np.c_[zeroS.T                         , zeroS.T, zeroS.T, zeroS.T@zeroV         , P                                             ]]


       RR = np.r_[B_cycle @ R,
                  B_cycle @ R,
                  zeroS,
                  zeroV.T @ zeroS,
                  R]

       # compute expectations
       # all of the A_matrices are already at k

       C_cycle, D_cycle = A_cycle, B_cycle
       C_trend, D_trend = A_trend, B_trend
       nexptot = nx*3*self.expectations
       TT_expectations = np.zeros((nexptot, TT.shape[0]))
       RR_expectations = np.zeros((nexptot, ns))


       value, shock = slice(3*nx,3*nx+nv), slice(3*nx+nv, TT.shape[0])

       total, tilde, bar = slice(0,nx),  slice(nx,2*nx), slice(2*nx,3*nx)
       itotal, itilde, ibar = slice(0,nx),  slice(nx,2*nx), slice(2*nx,3*nx)
       start=0
       for h in range(1, self.expectations+1):
           if h <= self.k:
               C_cycle = A_cycle_history[self.k - h] @ C_cycle
               D_cycle = A_cycle_history[self.k - h] @ D_cycle + B_cycle_history[self.k-h] @ np.linalg.matrix_power(P, h)

               C_trend = A_trend_history[self.k - h] @ C_trend
               D_trend = A_trend_history[self.k - h] @ D_trend + B_trend_history[self.k-h]
           else:
               C_cycle = A_cycle_history[0] @ C_cycle
               D_cycle = A_cycle_history[0] @ D_cycle + B_cycle_history[0] @ np.linalg.matrix_power(P, h)

               C_trend = A_trend_history[0] @ C_trend
               D_trend = A_trend_history[0] @ D_trend + B_trend_history[0]


           TT_expectations[itilde, tilde] = C_cycle
           TT_expectations[itilde, shock] = D_cycle @ P
           RR_expectations[itilde, :] = D_cycle

           TT_expectations[ibar, bar] = C_trend
           TT_expectations[ibar, value] = D_trend @ value_gammaC
           TT_expectations[ibar, total] = D_trend @ value_gamma @ value_Cx
           TT_expectations[ibar, shock] = D_trend @ value_gamma @ value_Cs

           TT_expectations[itotal, tilde] = C_cycle
           TT_expectations[itotal, shock] = D_cycle @ P + D_trend @ value_gamma @ value_Cs
           RR_expectations[itotal, :] = D_cycle

           TT_expectations[itotal, bar] = C_trend
           TT_expectations[itotal, value] = D_trend @ value_gammaC
           TT_expectations[itotal, total] = D_trend @ value_gamma @ value_Cx


           start += 3*nx
           itotal, itilde, ibar = slice(start,start+nx),  slice(start+nx,start+2*nx), slice(start+2*nx,start+3*nx)


       TT = np.r_[np.c_[TT, np.zeros((TT.shape[0], nexptot))],
                  np.c_[TT_expectations, np.zeros((nexptot, nexptot))]]
       RR = np.r_[RR,
                  RR_expectations]

       CC = np.zeros((TT.shape[0]))
       QQ = self.QQ(p0)
       DD = self.DD(p0)
       ZZ = self.ZZ(p0)
       HH = self.HH(p0)


       return CC, TT, RR, QQ, DD, ZZ, HH


class FHPRepAgent(Base):

    def __init___():
        pass

    def p0(self):
        return list(map(lambda x: self["calibration"]['parameters'][str(x)], self['parameters']))


    def smc(self, k=None, t0=0, expectations=None):
        k = self['k'] if k is None else k
        expectations = self['expectations'] if expectations is None else expectations
        cmodel = self.compile_model(k=k, expectations=expectations)
     
        npara = len(self['parameters'])
        para = sympy.IndexedBase("para", shape=(npara + 1,))
     
        from .symbols import Parameter
     
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
     
        context_tuple = ([(p.name, p) for p in self['parameters']]
        + [(p.name, p) for p in self["auxiliary_parameters"].keys()])
     
     
        context = dict(context_tuple)
        context["exp"] = sympy.exp
        context["log"] = sympy.log
     
        to_replace = {}
        for op, expr in self["auxiliary_parameters"].items():
            to_replace[op] = sympify(expr, context)
     
        to_replace = list(to_replace.items())
     
        from itertools import combinations, permutations
     
        edges = [
            (i, j)
            for i, j in permutations(to_replace, 2)
            if type(i[1]) not in [float, int] and i[1].has(j[0])
        ]
     
        from sympy import default_sort_key, topological_sort
     
        auxiliary_parameters = topological_sort([to_replace, edges], default_sort_key)
     
        system_matrices = self.system_matrices
        to_write = ['alpha0_cycle', 'alpha1_cycle', 'beta0_cycle',
                    'alphaC_cycle', 'alphaF_cycle', 'alphaB_cycle', 'betaS_cycle',
                    'alpha0_trend', 'alpha1_trend', 'betaV_trend',
                    'alphaC_trend', 'alphaF_trend', 'alphaB_trend',
                    'value_gammaC', 'value_gamma', 'value_Cx', 'value_Cs',
                    'P', 'R', 'self%QQ', 'DD2','self%ZZ','self%HH']
        print(len(system_matrices), len(to_write))
        fmats = [
            fcode(
                (mat.subs(auxiliary_parameters)).subs(fortran_subs),
                assign_to=n,
                source_format="free",
                standard=95,
                contract=False,
            )
            for mat, n in zip(system_matrices, to_write)
        ]
        sims_mat = "\n\n".join(fmats)

        # get templates/fhp.f90, making sure to use the correct pkg_resources filname
        from pkg_resources import resource_filename
        template_file = resource_filename("dsge", "templates/fhp.f90")
        with open(template_file, "r") as f:
            fortran_template = f.read()
            
        # turn template in f-string
        model = self
        p0 = ''
        system = sims_mat
        template = eval(f'f"""{fortran_template}"""')
        return template



    @classmethod
    def read(cls, model_yaml, k=None):
        dec = model_yaml['declarations']
        variables = [Variable(v) for v in dec['variables']]
        values = [Variable(v) for v in dec['values']]
        value_updates = [Variable(v) for v in dec['value_updates']]
        shocks = [Variable(v) for v in dec['shocks']]
        innovations = [Shock(v) for v in dec['innovations']]
        parameters = [Parameter(v) for v in dec['parameters']]
        expectations = dec['expectations'] if 'expectations' in dec else 0

        if "auxiliary_parameters" in dec:
            other_para = [Parameter(v) for v in dec["auxiliary_parameters"]]

            other_para = {op: sympify(model_yaml['calibration']["auxiliary_parameters"][op.name],
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
            obs_equations = {o: sympify(model_yaml["model"]["observables"][str(o)], context)
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
        
        # Check for future shocks in all equation groups
        from itertools import chain
        
        # Helper function moved outside the inner check_for_future_shocks function
        def find_shock_instances(eq, shock_list):
            """Find all instances of shocks in an equation."""
            return [atom for atom in eq.atoms() 
                    if isinstance(atom, Variable) and atom.name in [s.name for s in shock_list]]
        
        def get_original_equation(eq_idx, equation_type):
            """Get the original equation text from the YAML for better error messages."""
            # Parse the equation_type which might contain a path like 'cycle/terminal'
            parts = equation_type.split('/')
            
            if len(parts) == 2:
                # Handle nested sections like cycle/terminal, trend/plan, value/function
                section, subsection = parts
                if (section in yaml_eq and 
                    subsection in yaml_eq[section] and 
                    isinstance(yaml_eq[section][subsection], list) and 
                    eq_idx < len(yaml_eq[section][subsection])):
                    return yaml_eq[section][subsection][eq_idx]
            elif isinstance(yaml_eq.get(equation_type, []), list) and eq_idx < len(yaml_eq[equation_type]):
                # Handle flat sections like 'static'
                return yaml_eq[equation_type][eq_idx]
            
            return "unknown equation"
        
        def check_for_future_shocks(equation_list, shock_list, equation_type):
            """
            Check if any equation contains future-dated shocks, which are not allowed in FHP models.
            
            Args:
                equation_list: List of equations to check
                shock_list: List of shock variables to look for
                equation_type: Section of the model being checked (e.g., 'cycle/terminal')
            
            Raises:
                ValueError: If any future shock is found
            """                
            for eq_idx, eq in enumerate(equation_list):
                # Get all shock instances from both sides of the equation
                shock_instances = find_shock_instances(eq, shock_list) 
                
                # Check if any shock has a future date (date > 0)
                future_shocks = [s for s in shock_instances if s.date > 0]
                
                if future_shocks:
                    shock_names = set(s.name + "(" + str(s.date) + ")" for s in future_shocks)
                    original_eq = get_original_equation(eq_idx, equation_type)
                    
                    raise ValueError(
                        f"Future shocks are not allowed in FHP models. Found future shock(s) {', '.join(shock_names)} in "
                        f"equation: {original_eq} in section '{equation_type}'"
                    )
        
        # Check all equation types for future shocks
        # Check static equations
        if equations['static']:
            check_for_future_shocks(equations['static'], shocks, 'static')
            
        # Check cycle equations
        check_for_future_shocks(equations['cycle']['terminal'], shocks, 'cycle/terminal')
        check_for_future_shocks(equations['cycle']['plan'], shocks, 'cycle/plan')
            
        # Check trend equations
        check_for_future_shocks(equations['trend']['terminal'], shocks, 'trend/terminal')
        check_for_future_shocks(equations['trend']['plan'], shocks, 'trend/plan')
            
        # Check value function equations
        check_for_future_shocks(equations['value']['function'], shocks, 'value/function')
        check_for_future_shocks(equations['value']['update'], shocks, 'value/update')

        if 'covariance' in model_yaml['calibration']:
            QQ = from_dict_to_mat(model_yaml['calibration']['covariance'], innovations, context)
        else:
            print('No covariance matrix provided. Assuming identity matrix.')
            QQ = sympy.eye(len(innovations))

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
                      'expectations': expectations,
                      'innovations': innovations,
                      'parameters': parameters,
                      'auxiliary_parameters': other_para,
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

    def compile_model(self, k=None,expectations=None):

        k = self['k'] if k is None else k
        expectations = self['expectations'] if expectations is None else expectations

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


        all_para = self['parameters'] + list(self['auxiliary_parameters'].keys())

        lambdify_system_matrices = [lambdify(all_para, s)
                                    for s in system_matrices]

        intermediate_parameters = lambdify(self['parameters'], list(self['auxiliary_parameters'].values()))
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

        system_matrices.append(self['QQ'])
        QQ = expand_intermediate_parameters(lambdify(all_para, self['QQ']))
        nobs = len(self['observables'])
        all_obj = self['variables']+self['shocks']+self['innovations']+self['values']+self['value_updates']
        subs_dict = {}
        #subs_dict.update({v: 0 for v in all_obj})
        #subs_dict.update({v(-1): 0 for v in all_obj})
        #subs_dict.update({v(+j): 0 for v in all_obj for j in range(100)})
        subs_dict.update({v(+j): 0 for v in all_obj for j in range(-1,100)})

        DD = sympy.Matrix(nobs, 1, lambda i,j: self['obs_equations'][self['observables'][i]]
                          .subs(subs_dict))
        system_matrices.append(DD)
        DD = expand_intermediate_parameters(lambdify(all_para, DD))

        expectations_variables = []
        for j in range(1, expectations+1):
            expectations_variables += ([Variable(str(v))(j) for v in self['variables']]
                                       +[Variable(str(v)+'_cycle')(j) for v in self['variables']]
                                       +[Variable(str(v)+'_trend')(j) for v in self['variables']])


        ZZ_variables = (self['variables']
                        + [Variable(str(v)+'_cycle') for v in self['variables']]
                        + [Variable(str(v)+'_trend') for v in self['variables']]
                        + self['values'] + self['shocks'] + expectations_variables)
        ZZ = sympy.Matrix(nobs, len(ZZ_variables), lambda i, j: self['obs_equations'][self['observables'][i]].diff(ZZ_variables[j]))
        system_matrices.append(ZZ)
        ZZ = expand_intermediate_parameters(lambdify(all_para, ZZ))
        system_matrices.append(self['HH'])
        HH = expand_intermediate_parameters(lambdify(all_para, self['HH']))



        self.system_matrices = system_matrices

        prior = None
        from .Prior import Prior as pri
        if "prior" in self["estimation"]:
            prior = construct_prior(self["estimation"]["prior"], [str(pa) for pa in self['parameters']])
            prior = pri(prior)

        shock_names = [str(x) for x in self['innovations']]
        obs_names = [str(x) for x in self['observables']]
        state_names = ([str(x) for x in self['variables']]
                       + [str(x)+'_cycle' for x in self['variables']]
                     + [str(x)+'_trend' for x in self['variables']]
                     + [str(x) for x in self['values']]
                     + [str(x) for x in self['shocks']]
                     + [str(x) for x in expectations_variables]  )

        parameter_names = [str(x) for x in self['parameters']]

        linmod = LinearDSGEforFHPRepAgent(data, alpha0_cycle, alpha1_cycle,
                                          beta0_cycle, alphaC_cycle, alphaF_cycle, alphaB_cycle, betaS_cycle,
                                          alpha0_trend, alpha1_trend, betaV_trend, alphaC_trend, alphaF_trend,
                                          alphaB_trend, value_gammaC, value_gamma, value_Cx, value_Cs, P, R, QQ, DD, ZZ,
                                          HH, k, t0=0, expectations=expectations,
                                          shock_names=shock_names,
                                          state_names=state_names,
                                          obs_names=obs_names,
                                          prior=prior,
                                          parameter_names=parameter_names)
        return linmod
