#!/usr/bin/env python3
"""
DSGE - Dynamic Stochastic General Equilibrium model implementation.

This module provides the core DSGE model class, implementing the standard 
linear rational expectations framework.
"""

import numpy as np
import sympy
import logging
from typing import List, Dict, Union, Any, Optional, Tuple

from sympy.matrices import zeros
from sympy import sympify
from sympy.utilities.lambdify import lambdify

from .symbols import (Variable,
                     Equation,
                     Shock,
                     Parameter,
                     TSymbol,
                     reserved_names,
                     symbolic_context)

from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel, LinearDSGEModelwithSV
from .validation import validate_dsge_leads_lags, validate_model_consistency
from .logging_config import get_logger

from .parsing_tools import (parse_expression,
                           from_dict_to_mat,
                           parse_calibration,
                           construct_equation_list,
                           find_max_lead_lag)
from .Base import Base

# Get module logger
logger = get_logger("dsge.core")

class EquationList(list):
    """A container for holding a set of equations.

    Inherits the list class and includes additional argument fields for context.
    """

    def __init__(self, args: List[Equation], context: Dict[str, Union[int, float, TSymbol]] = {}):
        """
        Initialize an instance of the EquationList class.

        Args:
            args (list): A list of Equation objects.
            context (dict, optional): A dictionary containing the context for the equations. Defaults to {}.
        """
        self.context = context
        return super(EquationList, self).__init__(args)

    def __setitem__(self, key: int, value: Union[str, Equation]):
        if isinstance(value, str):
            lhs, rhs = value.split("=")

            lhs = eval(lhs, self.context)
            rhs = eval(rhs, self.context)

            value = Equation(lhs, rhs)

        return super(EquationList, self).__setitem__(key, value)


class DSGE(Base):

    max_lead = 1
    max_lag = 1

    numeric_context = {}

    def __init__(self, *kargs, **kwargs):
        """
        Initialize a DSGE model.
        
        Args:
            *kargs: Variable length list of arguments
            **kwargs: Arbitrary keyword arguments
            
        Raises:
            ValueError: If model validation fails
        """
        super(DSGE, self).__init__(self, *kargs, **kwargs)
        
        logger.info("Initializing DSGE model")
        
        # Validate model leads and lags
        if "equations" in self and "variables" in self:
            logger.debug("Validating model leads and lags")
            validation_errors = validate_dsge_leads_lags(
                self["equations"], 
                self["variables"],
                max_lead=self.max_lead,
                max_lag=self.max_lag
            )
            
            if validation_errors:
                for error in validation_errors:
                    logger.error(error)
                raise ValueError(
                    f"Model validation failed. The following errors were found:\n" + 
                    "\n".join(validation_errors)
                )
        
        # Model consistency checks (warning only)
        warnings = validate_model_consistency(self)
        for warning in warnings:
            logger.warning(warning)
        
        # Process variables
        fvars = []
        lvars = []

        # get forward looking variables
        logger.debug("Identifying forward and backward looking variables")
        for eq in self["equations"]:
            variable_too_far = [v for v in eq.atoms(Variable) if v.date > self.max_lead]
            variable_too_early = [v for v in eq.atoms(Variable) if v.date < -self.max_lag]

            if variable_too_far:
                logger.warning(f"Variables with leads beyond max_lead found: {variable_too_far}")
            
            if variable_too_early:
                logger.warning(f"Variables with lags beyond max_lag found: {variable_too_early}")

            eq_fvars = [v for v in eq.atoms(TSymbol) if v.date > 0]
            eq_lvars = [v for v in eq.atoms(TSymbol) if v.date < 0]

            for f in eq_fvars:
                if f not in fvars: fvars.append(f)

            for l in eq_lvars:
                if l not in lvars: lvars.append(l)
        
        logger.debug(f"Found {len(fvars)} forward-looking variables and {len(lvars)} backward-looking variables")
        
        self["info"]["nstate"] = len(self.variables) + len(fvars)

        self["fvars"] = fvars
        self["fvars_lagged"] = [Parameter("__LAGGED_" + f.name) for f in fvars]
        self["lvars"] = lvars
        self["re_errors"] = [Shock("eta_" + v.name) for v in self["fvars"]]

        # Create rational expectations errors equations
        logger.debug("Creating rational expectations error equations")
        self["re_errors_eq"] = []
        i = 0
        for fv, lag_fv in zip(fvars, self["fvars_lagged"]):
            self["re_errors_eq"].append(
                Equation(fv(-1) - lag_fv - self["re_errors"][i], sympify(0))
            )
            i += 1
            
        logger.info(f"DSGE model initialized with {len(self['variables'])} variables and {len(self['equations'])} equations")

        if "make_log" in self.keys():
            self["perturb_eq"] = []
            sub_dict = dict()
            sub_dict.update(
                {v: Parameter(v.name + "ss") * sympy.exp(v) for v in self["make_log"]}
            )
            sub_dict.update(
                {
                    v(-1): Parameter(v.name + "ss") * sympy.exp(v(-1))
                    for v in self["make_log"]
                }
            )
            sub_dict.update(
                {
                    v(1): Parameter(v.name + "ss") * sympy.exp(v(1))
                    for v in self["make_log"]
                }
            )

            for eq in self.equations:
                peq = eq.subs(sub_dict)
                self["perturb_eq"].append(peq)

            self["ss_ordering"] = [Variable(v.name + "ss") for v in self["make_log"]]

        else:
            self["perturb_eq"] = self["equations"]

        # context = [(s.name, s) for s in self['par_ordering']+self['var_ordering']+self['shk_ordering']+self['auxiliary_parameters']]
        # context = dict(context)
        # context['log'] = sympy.log
        # context['exp'] = sympy.exp
        context = {}
        # self['pertub_eq'] = EquationList(self['perturb_eq'], context)

        return

    def __repr__(self):
        indent = "\n    "
        repr = f"""
Model name: {self['name']}

Parameters: {self.parameters}

Variables: {self.variables}

Shocks: {self.shocks}

Equations:
        {indent.join([eq.__repr__() for eq in self.equations])}       
        """

        return repr

    


    @property
    def equations(self):
        return self["equations"]

    @property
    def variables(self):
        return self["var_ordering"]

    @property
    def shocks(self):
        return self["shk_ordering"]

    @property
    def neq(self):
        return len(self["perturb_eq"])

    @property
    def neq_fort(self):
        return self.neq + self.neta

    @property
    def neta(self):
        return len(self["fvars"])

    @property
    def ns(self):
        return

    @property
    def ny(self):
        return len(self["observables"])

    @property
    def neps(self):
        return len(self["shk_ordering"])

    @property
    def npara(self):
        return len(self.parameters)

    def p0(self):
        return list(map(lambda x: self["calibration"][str(x)], self.parameters))

    def python_sims_matrices(self, matrix_format="numeric"):

        vlist = self["var_ordering"] + self["fvars"]
        llist = [var(-1) for var in self["var_ordering"]] + self["fvars_lagged"]
        slist = self["shk_ordering"]

        subs_dict = dict()
        eq_cond = self["perturb_eq"] + self["re_errors_eq"]

        sub_var = self["var_ordering"]
        subs_dict.update({v: 0 for v in sub_var})
        subs_dict.update({v(1): 0 for v in sub_var})
        subs_dict.update({v(-1): 0 for v in sub_var})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self["re_errors"])
        ovar = len(self["observables"])

        GAM0 = zeros(svar, svar)
        GAM1 = zeros(svar, svar)
        PSI = zeros(svar, evar)
        PPI = zeros(svar, rvar)


        for eq_i, eq in enumerate(eq_cond):
            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))
            #print('---------------------------------------')
            #print(f'Equation: {eq}')
            #print(f'current variables {list(curr_var)}')
            for v in curr_var:
                #print(f'\tdifferentiating wrt to {v}')
                v_j = vlist.index(v)
                GAM0[eq_i, v_j] = -(eq).set_eq_zero.diff(v).subs(subs_dict)

            past_var = filter(lambda x: x in llist, eq.atoms())
            #print(f'past variables: {list(past_var)}.')
            for v in past_var:
                deq_dv = eq.set_eq_zero.diff(v).subs(subs_dict)
                v_j = llist.index(v)
                GAM1[eq_i, v_j] = deq_dv

                #print(f'\tdifferentiating wrt to {v}')

            #print(f'GAM1: {GAM1[eq_i,:]}')

            shocks = filter(lambda x: x, eq.atoms(Shock))

            for s in shocks:
                if s not in self["re_errors"]:
                    s_j = slist.index(s)
                    PSI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)
                else:
                    s_j = self["re_errors"].index(s)
                    PPI[eq_i, s_j] = eq.set_eq_zero.diff(s).subs(subs_dict)


            # print "\r Differentiating equation {0} of {1}.".format(eq_i, len(eq_cond)),
        DD = zeros(ovar, 1)
        ZZ = zeros(ovar, svar)

        eq_i = 0
        for obs in self["observables"]:
            eq = self["obs_equations"][str(obs)]

            DD[eq_i, 0] = eq.subs(subs_dict)

            curr_var = filter(lambda x: x.date >= 0, eq.atoms(Variable))
            for v in curr_var:
                v_j = vlist.index(v)
                ZZ[eq_i, v_j] = eq.diff(v).subs(subs_dict)

            eq_i += 1

        if matrix_format == "symbolic":
            QQ = self["covariance"]
            HH = self["measurement_errors"]
            return GAM0, GAM1, PSI, PPI, QQ, DD, ZZ, HH

        subs_dict = []
        context = dict([(p, Parameter(p)) for p in self.parameters])
        context.update(reserved_names)
        context['normcdf'] = lambda x: 0.5 * (1 + sympy.erf(x / sympy.sqrt(2)))
        context_f = {}
        context_f["exp"] = np.exp

        if "external" in self["__data__"]["declarations"]:
            from importlib.machinery import SourceFileLoader
            import importlib.util
         
            f = self["__data__"]["declarations"]["external"]["file"]
            
            # Load module from source file
            spec = importlib.util.spec_from_loader("external", SourceFileLoader("external", f))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for n in self["__data__"]["declarations"]["external"]["names"]:
                context[n] = sympy.Function(n)  # or getattr(module, n) for other uses, if necessary
                context_f[n] = getattr(module, n)

        self.GAM0 = self.lambdify(GAM0, context=context_f) 
        self.GAM1 = self.lambdify(GAM1, context=context_f) 
        self.PSI = self.lambdify(PSI, context=context_f)
        self.PPI = self.lambdify(PPI, context=context_f)

        self.QQ = self.lambdify(self["covariance"], context=context_f)
        self.HH = self.lambdify(self["measurement_errors"], context=context_f)

        self.DD = self.lambdify(DD, context=context_f)
        self.ZZ = self.lambdify(ZZ, context=context_f)

        self.psi = None

        # if self['__data__']['declarations']['type'] == 'sv':
        #     sv = self['__data__']['equations']['sv']
        #     p0 = self.p0()
        #     QQ_obs = self.QQ(p0)
        #     nshocks = len(self.shocks)
        #     assert (QQ_obs == np.eye(nshocks)).all()
        #
        #     Lambda = from_dict_to_mat(sv['transition'], self['shk_ordering'], context)
        #     Omega = from_dict_to_mat(sv['covariance'], self['shk_ordering'], context)
        #     Omega0 = from_dict_to_mat(sv['initial_covariance'], self['shk_ordering'], context)
        #
        #     self.Lambda = add_auxiliary_parameters(lambdify(all_para, Lambda))
        #     self.Omega = add_auxiliary_parameters(lambdify(all_para, Omega))
        #     self.Omega0 = add_auxiliary_parameters(lambdify(all_para, Omega0))

        return GAM0, GAM1, PSI, PPI

    def compile_model(self):
        self.python_sims_matrices()

        GAM0 = self.GAM0
        GAM1 = self.GAM1
        PSI = self.PSI
        PPI = self.PPI

        QQ = self.QQ
        DD = self.DD
        ZZ = self.ZZ
        HH = self.HH

        if "observables" not in self:
            self["observables"] = self["variables"].copy()
            self["obs_equations"] = dict(self["observables"], self["observables"])

        if "data" in self["__data__"]["estimation"]:
            data = read_data_file(
                self["__data__"]["estimation"]["data"], self["observables"]
            )
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        prior = None
        if "prior" in self["__data__"]["estimation"]:
            prior = construct_prior(
                self["__data__"]["estimation"]["prior"], self.parameters
            )

        from .Prior import Prior as pri

        if 1==0:#self['__data__']['declarations']['type'] == 'sv':

            dsge = LinearDSGEModelwithSV(
                  data,
                  GAM0,
                  GAM1,
                  PSI,
                  PPI,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                self.Lambda,
                self.Omega,
                self.Omega0,
                  t0=0,
                  shock_names=list(map(str, self.shocks)),
                  state_names=list(map(str, self.variables + self["fvars"])),
                  obs_names=list(map(str, self["observables"])),
                  prior=pri(prior),
                  parameter_names=self.parameters
              )
        else:
            dsge = LinearDSGEModel(
                  data,
                  GAM0,
                  GAM1,
                  PSI,
                  PPI,
                  QQ,
                  DD,
                  ZZ,
                  HH,
                  t0=0,
                  shock_names=list(map(str, self.shocks)),
                  state_names=list(map(str, self.variables + self["fvars"])),
                  obs_names=list(map(str, self["observables"])),
                  prior=pri(prior),
                  parameter_names=self.parameters
              )

            dsge.psi = self.psi

            return dsge

    def update_data_file(self, file_path, start_date=None):
        if start_date is None:
            self['__data__']['estimation']['data'] = file_path
        else:
            self['__data__']['estimation']['data'] = {'file': file_path, 'start': start_date}

        return None

    def solve_model(self, p0):

        # if self.GAM0 == None:
        self.python_sims_matrices()

        from gensys import gensys_wrapper as gensys

        G0 = self.GAM0(*p0)
        G1 = self.GAM1(*p0)
        PSI = self.PSI(*p0)
        PPI = self.PPI(*p0)
        C0 = np.zeros((G0.shape[0]))
        TT, CC, RR, fmat, fwt, ywt, gev, RC, loose = gensys.call_gensys(
            G0, G1, C0, PSI, PPI, 1.00000000001
        )

        return TT, RR, RC

    @classmethod
    def read(cls, model_yaml):

        dec, cal = model_yaml["declarations"], model_yaml["calibration"]
        name = dec["name"]

        var_ordering = [Variable(v) for v in dec["variables"]]
        par_ordering = [Parameter(v) for v in dec["parameters"]]
        shk_ordering = [Shock(v) for v in dec["shocks"]]

        if "auxiliary_parameters" in dec:
            other_para = [Parameter(v) for v in dec["auxiliary_parameters"]]
        else:
            other_para = []

        if "observables" in dec:
            observables = [Variable(v) for v in dec["observables"]]
            obs_equations = model_yaml["equations"]["observables"]
        else:
            observables = [Variable(v) for v in dec["variables"]]
            obs_equations = {v: v for v in dec['variables']}

        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
        else:
            measurement_errors = None

        if "make_log" in dec:
            make_log = [Variable(v) for v in dec["make_log"]]
        else:
            make_log = []

        steady_state = [0]
        init_values = [0]

        context = {s.name: s
                   for s in var_ordering + par_ordering + shk_ordering + other_para}
        context.update(symbolic_context)

        if "model" in model_yaml["equations"]:
            raw_equations = model_yaml["equations"]["model"]
        else:
            raw_equations = model_yaml["equations"]

        equations = construct_equation_list(raw_equations, context)


        # ------------------------------------------------------------
        # Figure out max leads and lags
        # ------------------------------------------------------------
        (max_lead_exo,
         max_lag_exo) = find_max_lead_lag(equations, shk_ordering)

        # arbitrary lags of exogenous shocks
        for s in shk_ordering:
            if abs(max_lag_exo[s]) > 0:
                var_s = Variable(s.name + "_VAR")
                var_ordering.append(var_s)
                equations.append(Equation(var_s, s))

                subs1 = [s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs2 = [var_s(-i) for i in np.arange(1, abs(max_lag_exo[s]) + 1)]
                subs_dict = dict(zip(subs1, subs2))
                equations = [eq.subs(subs_dict) for eq in equations]

        (max_lead_endo,
         max_lag_endo) = find_max_lead_lag(equations, var_ordering)


        # ------------------------------------------------------------
        # arbitrary lags/leads of endogenous variables
        # ------------------------------------------------------------
        subs_dict = dict()
        old_var = var_ordering[:]
        for v in old_var:
            # lags
            for i in np.arange(2, abs(max_lag_endo[v]) + 1):
                # for lag l need to add l-1 variable
                var_l = Variable(v.name + "_LAG" + str(i - 1))

                if i == 2:
                    var_l_1 = Variable(v.name, date=-1)
                else:
                    var_l_1 = Variable(v.name + "_LAG" + str(i - 2), date=-1)

                subs_dict[Variable(v.name, date=-i)] = var_l(-1)
                var_ordering.append(var_l)
                equations.append(Equation(var_l, var_l_1))

            # still need to do leads

        equations = [eq.subs(subs_dict) for eq in equations]
        if 'covariance' in cal:
            QQ = from_dict_to_mat(cal['covariance'], shk_ordering, context)
        else:
            print('No covariance matrix provided. Assuming identity matrix.')
            QQ = sympy.eye(len(shk_ordering))

        #------------------------------------------------------------------
        # observation equation
        #------------------------------------------------------------------
        context["sum"] = np.sum
        context["range"] = range
        for obs in obs_equations.items():
            obs_equations[obs[0]] = eval(obs[1], context)

        me_dict = {}
        if 'measurement_errors' in model_yaml['calibration']:
            me_dict = model_yaml['calibration']['measurement_errors']

        if measurement_errors is not None:
            HH = from_dict_to_mat(me_dict, measurement_errors, context)
        else:
            HH = from_dict_to_mat(me_dict, observables, context)
        
        calibration = model_yaml["calibration"]["parameters"]

        if "auxiliary_parameters" not in cal:
            cal["auxiliary_parameters"] = {}
        else:
            cal['auxiliary_parameters'] = {op:
                                           parse_expression(cal["auxiliary_parameters"][str(op)],
                                                         {str(p): p for p in
                                                          par_ordering+other_para})

                                           for op in other_para}


        if 'estimation' not in model_yaml:
            model_yaml['estimation'] = {}
            
        model_dict = {
            "var_ordering": var_ordering,
            "parameters": par_ordering,
            "shk_ordering": shk_ordering,
            "other_parameters": other_para,
            "other_para": other_para,
            "auxiliary_parameters": cal["auxiliary_parameters"],
            "calibration": calibration,
            "steady_state": steady_state,
            "init_values": init_values,
            "equations": equations,
            "covariance": QQ,
            "measurement_errors": HH,
            "meas_ordering": measurement_errors,
            "info": dict(),
            "make_log": make_log,
            "__data__": model_yaml,
            "name": dec["name"],
            "observables": observables,
            "obs_equations": obs_equations,
        }

        model = cls(**model_dict)
        return model
