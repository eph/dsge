#!/usr/bin/env python3
"""
DSGE - Dynamic Stochastic General Equilibrium model implementation.

This module provides the core DSGE model class, implementing the standard 
linear rational expectations framework.
"""

import numpy as np
import sympy
from typing import List, Dict, Union

from sympy.matrices import zeros
from sympy import sympify

from .symbols import (Variable,
                     Equation,
                     Shock,
                     Parameter,
                     TSymbol,
                     reserved_names)

from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel
from .validation import validate_dsge_leads_lags, validate_model_consistency
from .logging_config import get_logger

from .parsing_tools import (parse_expression,
                            from_dict_to_mat,
                            construct_equation_list,
                            find_max_lead_lag,
                            build_symbolic_context)
from .Base import Base

# Get module logger
logger = get_logger("dsge.core")

class EquationList(list):
    """A container for holding a set of equations.

    Inherits the list class and includes additional argument fields for context.
    """

    def __init__(
        self,
        args: List[Equation],
        context: Dict[str, Union[int, float, TSymbol]] | None = None,
    ):
        """
        Initialize an instance of the EquationList class.

        Args:
            args (list): A list of Equation objects.
            context (dict, optional): A dictionary containing the context for the equations. Defaults to {}.
        """
        self.context = {} if context is None else context
        return super(EquationList, self).__init__(args)

    def __setitem__(self, key: int, value: Union[str, Equation]):
        if isinstance(value, str):
            lhs, rhs = value.split("=")

            # Use safe parser instead of eval
            lhs = parse_expression(lhs, self.context)
            rhs = parse_expression(rhs, self.context)

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

        # Prefer per-model settings (from YAML declarations) over class defaults.
        if "max_lead" in self:
            self.max_lead = int(self["max_lead"])
        if "max_lag" in self:
            self.max_lag = int(self["max_lag"])
        
        logger.info("Initializing DSGE model")
        
        # Validate model leads and lags
        if "equations" in self and "var_ordering" in self:
            logger.debug("Validating model leads and lags")
            validation_errors = validate_dsge_leads_lags(
                self["equations"], 
                self["var_ordering"],
                max_lead=self.max_lead,
                max_lag=self.max_lag
            )
            
            if validation_errors:
                for error in validation_errors:
                    logger.error(error)
                raise ValueError(
                    "Model validation failed. The following errors were found:\n" + 
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
            # For large models, repeated `atoms(...)` calls are a major startup cost.
            # Compute the time-indexed symbols once and reuse.
            ts_atoms = eq.atoms(TSymbol)
            var_atoms = [v for v in ts_atoms if isinstance(v, Variable)]

            variable_too_far = [v for v in var_atoms if v.date > self.max_lead]
            variable_too_early = [v for v in var_atoms if v.date < -self.max_lag]

            if variable_too_far:
                logger.warning(f"Variables with leads beyond max_lead found: {variable_too_far}")
            
            if variable_too_early:
                logger.warning(f"Variables with lags beyond max_lag found: {variable_too_early}")

            eq_fvars = [v for v in ts_atoms if v.date > 0]
            eq_lvars = [v for v in ts_atoms if v.date < 0]

            for f in eq_fvars:
                if f not in fvars:
                    fvars.append(f)

            for lagged_var in eq_lvars:
                if lagged_var not in lvars:
                    lvars.append(lagged_var)
        
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
            
        # Get the number of variables and equations from the model itself
        num_vars = len(self["var_ordering"]) if "var_ordering" in self else 0
        num_eqs = len(self["equations"]) if "equations" in self else 0
        logger.info(f"DSGE model initialized with {num_vars} variables and {num_eqs} equations")

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
        return list(map(lambda x: self["calibration"][x], self.parameters))

    def python_sims_matrices(
        self,
        matrix_format="numeric",
        method: str = "auto",
        lre_form: str = "auto",
    ):
        """
        Construct Sims-style system matrices for the linear rational expectations model.

        Notes
        -----
        For historical reasons, the default ("legacy") representation used a mismatched
        `(vlist, llist)` pair (lead symbols in `vlist` but `__LAGGED_*` placeholders in `llist`).
        On large models with auxiliary lead variables this can produce a singular pencil
        (GENSYS "coincident zeros").

        The `"expvars"` form replaces each lead symbol `x(+1)` with a distinct current-period
        expectation variable, so the lag list is the true lag of the current list.
        """
        lre_form = str(lre_form).lower().strip()
        if lre_form not in {"auto", "legacy", "expvars"}:
            raise ValueError("lre_form must be one of {'auto','legacy','expvars'}.")

        # Heuristic: the legacy LRE augmentation uses an inconsistent (vlist,llist) pair where
        # G0 columns correspond to `fvars` but G1 columns correspond to `__LAGGED_*` placeholders.
        # This can produce singular pencils on large models with auxiliary lead variables (e.g. LINVER).
        if lre_form == "auto":
            # If the YAML declared max_lead > 1, we almost surely introduced Dynare-style lead auxiliaries.
            # Use the more robust expvars form.
            lre_form = "expvars" if int(getattr(self, "max_lead", 1)) > 1 else "legacy"

        if matrix_format != "symbolic":
            already = (
                callable(getattr(self, "GAM0", None))
                and callable(getattr(self, "GAM1", None))
                and callable(getattr(self, "PSI", None))
                and callable(getattr(self, "PPI", None))
                and callable(getattr(self, "QQ", None))
                and callable(getattr(self, "HH", None))
                and callable(getattr(self, "DD", None))
                and callable(getattr(self, "ZZ", None))
            )
            if already and getattr(self, "_python_sims_lre_form", None) == lre_form:
                return None

        if lre_form == "legacy":
            vlist = self["var_ordering"] + self["fvars"]
            llist = [var(-1) for var in self["var_ordering"]] + self["fvars_lagged"]
            slist = self["shk_ordering"]
            eq_cond = self["perturb_eq"] + self["re_errors_eq"]
        else:
            # Robust "expectation variables" formulation:
            # - replace each lead symbol x(+1) with a distinct current-period symbol E_x
            # - build lags consistently as E_x(-1), so `[G0;G1]` is a proper matrix pencil
            fvars = list(self.get("fvars", []))
            if any(not isinstance(v, Variable) for v in fvars):
                raise NotImplementedError(
                    "lre_form='expvars' currently supports only endogenous lead variables (no shock leads). "
                    "Use lre_form='legacy' for models with anticipated shocks written as eps(+k)."
                )

            used_names = {v.name for v in self["var_ordering"]}
            exp_vars = []
            repl = {}
            for fv in fvars:
                base = f"__E_{fv.name}"
                name = base
                j = 1
                while name in used_names:
                    j += 1
                    name = f"{base}_{j}"
                used_names.add(name)
                ev = Variable(name)
                exp_vars.append(ev)
                repl[fv] = ev

            vlist = self["var_ordering"] + exp_vars
            llist = [var(-1) for var in self["var_ordering"]] + [ev(-1) for ev in exp_vars]
            slist = self["shk_ordering"]

            eq_model = [eq.xreplace(repl) for eq in self["perturb_eq"]]
            # Forecast error equations: x_t - E_{t-1}[x_t] - eta_t = 0
            eq_re = [
                Equation(fv(-1) - ev(-1) - eta, sympify(0))
                for fv, ev, eta in zip(fvars, exp_vars, self["re_errors"])
            ]
            eq_cond = eq_model + eq_re

        vpos = {v: i for i, v in enumerate(vlist)}
        lpos = {v: i for i, v in enumerate(llist)}
        spos = {s: i for i, s in enumerate(slist)}
        rpos = {s: i for i, s in enumerate(self["re_errors"])}

        def _subs_steady_state(expr):
            """Evaluate an expression at the (zero) steady state for endo/shocks only."""
            atoms = expr.atoms(Variable) | expr.atoms(Shock)
            if not atoms:
                return expr
            return expr.subs({a: 0 for a in atoms})

        svar = len(vlist)
        evar = len(slist)
        rvar = len(self["re_errors"])
        ovar = len(self["observables"])

        GAM0 = zeros(svar, svar)
        GAM1 = zeros(svar, svar)
        PSI = zeros(svar, evar)
        PPI = zeros(svar, rvar)


        method = str(method).lower()
        if method not in {"auto", "jacobian", "jac", "loop", "sparse"}:
            # Backward-compatible behavior: unknown methods behave like the loop backend.
            method = "loop"

        # Heuristic: SymPy's full-system `.jacobian(...)` can be much slower than sparse/loop
        # differentiation on large, sparse models (e.g. linver). Keep Jacobian for small systems.
        if method == "auto":
            n_eq = len(eq_cond)
            n_v = len(vlist)
            n_l = len(llist)
            n_s = len(slist)
            n_r = len(self["re_errors"])
            # Rough proxy for expression/Jacobian size (not counting algebraic complexity).
            size_proxy = n_eq * (n_v + n_l + n_s + n_r)
            use_jacobian = (n_eq <= 150) and (size_proxy <= 150_000)
        else:
            use_jacobian = method in {"jacobian", "jac"}

        if use_jacobian:
            try:
                res = sympy.Matrix([eq.set_eq_zero for eq in eq_cond])
                subs_ss = {a: 0 for a in (res.atoms(Variable) | res.atoms(Shock))}

                GAM0 = (-res.jacobian(vlist)).subs(subs_ss)
                GAM1 = (res.jacobian(llist)).subs(subs_ss)
                PSI = (res.jacobian(slist)).subs(subs_ss) if evar else zeros(svar, 0)
                PPI = (res.jacobian(self["re_errors"])).subs(subs_ss) if rvar else zeros(svar, 0)
            except Exception as e:
                if method in {"jacobian", "jac"}:
                    raise
                logger.warning(f"Falling back to loop differentiation in python_sims_matrices: {e}")
                use_jacobian = False

        if not use_jacobian:
            for eq_i, eq in enumerate(eq_cond):
                eq0 = eq.set_eq_zero
                var_atoms = eq0.atoms(Variable)
                shock_atoms = eq0.atoms(Shock)
                subs_atoms = var_atoms | shock_atoms
                subs0 = {a: 0 for a in subs_atoms} if subs_atoms else None

                def _subs_steady_state_fast(expr):
                    if subs0 is None:
                        return expr
                    return expr.subs(subs0)

                curr_var = [v for v in var_atoms if v.date >= 0 and v in vpos]
                for v in curr_var:
                    v_j = vpos[v]
                    GAM0[eq_i, v_j] = -_subs_steady_state_fast(eq0.diff(v))

                past_var = [v for v in var_atoms if v in lpos]
                for v in past_var:
                    deq_dv = _subs_steady_state_fast(eq0.diff(v))
                    v_j = lpos[v]
                    GAM1[eq_i, v_j] = deq_dv

                for s in shock_atoms:
                    if s not in self["re_errors"]:
                        s_j = spos[s]
                        PSI[eq_i, s_j] = _subs_steady_state_fast(eq0.diff(s))
                    else:
                        s_j = rpos[s]
                        PPI[eq_i, s_j] = _subs_steady_state_fast(eq0.diff(s))

                # print "\r Differentiating equation {0} of {1}.".format(eq_i, len(eq_cond)),
        DD = zeros(ovar, 1)
        ZZ = zeros(ovar, svar)

        if use_jacobian:
            obs_exprs = [self["obs_equations"][obs.name] for obs in self["observables"]]
            obs_res = sympy.Matrix(obs_exprs)
            subs_ss_obs = {a: 0 for a in (obs_res.atoms(Variable) | obs_res.atoms(Shock))}
            DD = obs_res.subs(subs_ss_obs)
            ZZ = (obs_res.jacobian(vlist)).subs(subs_ss_obs)
        else:
            eq_i = 0
            for obs in self["observables"]:
                eq = self["obs_equations"][obs.name]

                atoms = eq.atoms(Variable) | eq.atoms(Shock)
                subs0 = {a: 0 for a in atoms} if atoms else None

                def _subs_steady_state_obs(expr):
                    if subs0 is None:
                        return expr
                    return expr.subs(subs0)

                DD[eq_i, 0] = _subs_steady_state_obs(eq)

                var_atoms = eq.atoms(Variable)
                curr_var = [v for v in var_atoms if v.date >= 0 and v in vpos]
                for v in curr_var:
                    v_j = vpos[v]
                    ZZ[eq_i, v_j] = _subs_steady_state_obs(eq.diff(v))

                eq_i += 1

        if matrix_format == "symbolic":
            QQ = self["covariance"]
            HH = self["measurement_errors"]
            self._python_sims_lre_form = lre_form
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
        self._python_sims_lre_form = lre_form

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

    def compile_model(
        self,
        order=1,
        pruning=True,
        nonlinear_observables: str = "error",
        lre_reduction: str = "none",
        lre_form: str = "auto",
    ):
        """
        Compile the DSGE model into an object with likelihood/simulation APIs.

        Parameters
        ----------
        order : int
            Perturbation order. `order=1` returns the existing linear-Gaussian
            state-space object (Kalman filter). `order=2` returns an order-2
            pruned perturbation model (particle filter likelihood).
        pruning : bool
            For `order=2`, use Dynare-style pruning for simulation/filtering.
        nonlinear_observables : {"error", "linearize"}
            Only used when `order=2`. If `"error"` (default), observable equations must
            be affine in current-period endogenous variables. If `"linearize"`, allow
            nonlinear observables and interpret them via a first-order linearization
            at the steady state (still disallow lags/leads and shocks in observables).
        lre_form : {"auto", "legacy", "expvars"}
            Only used when `order=1`. Controls the internal LRE augmentation used to
            construct the GENSYS pencil. `"auto"` selects `"expvars"` for models with
            declared `max_lead > 1` (e.g. LINVER) and `"legacy"` otherwise.
        """
        if order == 2:
            from .perturbation_model import PerturbationDSGEModel

            if "observables" not in self:
                self["observables"] = self["variables"].copy()
                self["obs_equations"] = dict(self["observables"], self["observables"])

            if "data" in self["estimation"]:
                data = read_data_file(
                    self["estimation"]["data"], self["observables"]
                )
            else:
                data = np.nan * np.ones((100, len(self["observables"])))

            prior = None
            if "prior" in self["estimation"]:
                prior = construct_prior(
                    self["estimation"]["prior"], self.parameters
                )

            from .Prior import Prior as pri

            return PerturbationDSGEModel(
                dsge_model=self,
                yy=data,
                t0=0,
                shock_names=list(map(str, self.shocks)),
                state_names=None,
                obs_names=list(map(str, self["observables"])),
                prior=pri(prior),
                parameter_names=self.parameters,
                order=2,
                pruning=pruning,
                nonlinear_observables=nonlinear_observables,
            )

        if order != 1:
            raise ValueError(f"Unsupported perturbation order: {order}. Use order=1 or order=2.")

        lre_reduction = str(lre_reduction).lower().strip()
        if lre_reduction not in {"none", "core"}:
            raise ValueError("lre_reduction must be one of {'none','core'}.")
        if lre_reduction == "core":
            core = self._lre_core_model()
            # Important: avoid recursive reduction.
            return core.compile_model(order=1, lre_reduction="none", lre_form=lre_form)

        self.python_sims_matrices(lre_form=lre_form)

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

        if "data" in self["estimation"]:
            data = read_data_file(
                self["estimation"]["data"], self["observables"]
            )
        else:
            data = np.nan * np.ones((100, len(self["observables"])))

        prior = None
        if "prior" in self["estimation"]:
            prior = construct_prior(
                self["estimation"]["prior"], self.parameters
            )

        from .Prior import Prior as pri

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
            parameter_names=self.parameters,
        )

        dsge.psi = self.psi

        return dsge

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

    def solve_second_order(self, p0=None, **kwargs):
        """
        Compute a second-order perturbation solution for the model (LRE only).

        Returns
        -------
        dsge.second_order.SecondOrderSolution
        """
        if p0 is None:
            p0 = self.p0()
        from .second_order import solve_second_order

        return solve_second_order(self, p0, **kwargs)

    def dynare_first_order_solution(self, *, p0=None, timeout: int = 240):
        """
        Solve the model with Dynare (order=1) and return Dynare's first-order objects.

        This is intended as a robustness cross-check for large models when our
        internal `gensys` plumbing flags coincident zeros / singular pencils.
        """
        if p0 is None:
            p0 = self.p0()

        # Ensure any matrix construction is done (and that auxiliary parameters are evaluated)
        # so Dynare export sees the same calibrated parameter values.
        _ = p0

        from .dynare_export import to_dynare_mod
        from .dynare_integration import load_first_order_solution, run_dynare_mod_text

        mod = to_dynare_mod(self, order=1, pruning=False, irf=0, periods=0)
        name = str(self.get("name", "dsge_model"))
        safe_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
        results_path = run_dynare_mod_text(mod_text=mod.mod_text, model_name=safe_name, timeout=int(timeout))
        return load_first_order_solution(results_path)

    @staticmethod
    def _shift_expr(expr, shift: int):
        """Shift all time-indexed symbols in an expression by `shift` periods."""
        if shift == 0:
            return expr
        subs = {}
        for v in expr.atoms(Variable):
            subs[v] = Variable(v.name, date=v.date + shift)
        for s in expr.atoms(Shock):
            subs[s] = Shock(s.name, date=s.date + shift)
        return expr.xreplace(subs)

    def _lre_core_model(self):
        """
        Build a reduced 'core' DSGE model intended for solving large LRE systems.

        The core model:
        - removes dead-end variables whose defining equations contain leads and are
          otherwise unused (typical 'reporting' / auxiliary forward variables), and
        - performs safe one-step substitutions for orphan lead variables when their
          defining equations are purely backward-looking (no leads, no shocks), and
          the substitution does not introduce new orphan leads.

        This is designed to address singular matrix pencils (coincident zeros) in
        very large models that include unused forward-looking reporting variables.
        """
        var_ordering = list(self["var_ordering"])
        equations = list(self["equations"])

        # ------------------------------------------------------------
        # Pass 1: safe orphan-lead substitution (iterative)
        # ------------------------------------------------------------
        lhs_eq = {
            eq.lhs.name: eq
            for eq in equations
            if isinstance(eq.lhs, Variable) and getattr(eq.lhs, "date", 0) == 0
        }

        def lead_counts(eqs):
            counts = {}
            for eq in eqs:
                for v in eq.set_eq_zero.atoms(Variable):
                    if v.date == 1:
                        counts[v] = counts.get(v, 0) + 1
            return counts

        changed = True
        # Limit iterations to avoid pathological loops.
        for _ in range(10):
            if not changed:
                break
            changed = False
            counts = lead_counts(equations)
            # Identify orphan lead variables that appear in exactly one equation.
            orphan_leads = [v for v, c in counts.items() if c == 1]
            for v1 in orphan_leads:
                base = v1.name
                if base not in lhs_eq:
                    continue
                base_eq = lhs_eq[base]
                rhs = base_eq.rhs
                # Only substitute leads implied by purely backward, shock-free equations.
                if rhs.atoms(Shock):
                    continue
                rhs_vars = rhs.atoms(Variable)
                if rhs_vars and max(v.date for v in rhs_vars) > 0:
                    continue

                rhs_shift = self._shift_expr(rhs, 1)
                # Guard: don't introduce new orphan leads.
                new_leads = [v for v in rhs_shift.atoms(Variable) if v.date == 1]
                introduces_orphans = any(counts.get(v, 0) <= 1 and v != v1 for v in new_leads)
                if introduces_orphans:
                    continue

                repl = {v1: rhs_shift}
                equations = [eq.xreplace(repl) for eq in equations]
                lhs_eq = {
                    eq.lhs.name: eq
                    for eq in equations
                    if isinstance(eq.lhs, Variable) and getattr(eq.lhs, "date", 0) == 0
                }
                changed = True

        # ------------------------------------------------------------
        # Pass 2: prune dead-end forward reporting variables (iterative)
        # ------------------------------------------------------------
        while True:
            # Map variable name -> set of equation indices where it appears (any lag/lead).
            occ = {}
            for i, eq in enumerate(equations):
                for v in eq.set_eq_zero.atoms(Variable):
                    occ.setdefault(v.name, set()).add(i)

            lhs_index = {
                eq.lhs.name: i
                for i, eq in enumerate(equations)
                if isinstance(eq.lhs, Variable) and getattr(eq.lhs, "date", 0) == 0
            }

            drop_names = []
            for v in var_ordering:
                name = v.name
                if name not in lhs_index:
                    continue
                idx = lhs_index[name]
                if occ.get(name, set()) != {idx}:
                    continue
                # Only drop if its defining equation contains a lead.
                has_lead = any(
                    vv.date > 0 for vv in equations[idx].set_eq_zero.atoms(Variable)
                )
                if has_lead:
                    drop_names.append(name)

            if not drop_names:
                break

            drop_set = set(drop_names)
            var_ordering = [v for v in var_ordering if v.name not in drop_set]
            equations = [
                eq
                for eq in equations
                if not (isinstance(eq.lhs, Variable) and eq.lhs.name in drop_set and eq.lhs.date == 0)
            ]

        keep_names = {v.name for v in var_ordering}

        # Observables: if user explicitly provided observables, require they all survive.
        explicit_obs = False
        try:
            explicit_obs = "observables" in self["__data__"]["declarations"]
        except Exception:
            explicit_obs = False

        if explicit_obs:
            missing = [v.name for v in self["observables"] if v.name not in keep_names]
            if missing:
                raise ValueError(
                    "lre_reduction='core' dropped variables that are declared as observables: "
                    + ", ".join(missing[:20])
                )
            observables = list(self["observables"])
            obs_equations = {k: v for k, v in self["obs_equations"].items() if k in {o.name for o in observables}}
        else:
            observables = [Variable(v.name) for v in var_ordering]
            obs_equations = {v.name: Variable(v.name) for v in observables}

        model_dict = {
            "var_ordering": var_ordering,
            "parameters": self["parameters"],
            "shk_ordering": self["shk_ordering"],
            "other_parameters": self.get("other_parameters", []),
            "other_para": self.get("other_para", []),
            "auxiliary_parameters": self.get("auxiliary_parameters", {}),
            "calibration": self["calibration"],
            "steady_state": self.get("steady_state", [0]),
            "init_values": self.get("init_values", [0]),
            "equations": equations,
            "covariance": self["covariance"],
            "measurement_errors": self["measurement_errors"],
            "meas_ordering": self.get("meas_ordering", None),
            "info": dict(),
            "make_log": self.get("make_log", []),
            "estimation": self.get("estimation", {}),
            "__data__": self.get("__data__", {}),
            "name": f"{self.get('name','model')}_core",
            "observables": observables,
            "obs_equations": obs_equations,
            "max_lead": int(getattr(self, "max_lead", 1)),
            "max_lag": int(getattr(self, "max_lag", 1)),
        }
        return DSGE(**model_dict)

    @classmethod
    def read(cls, model_yaml):

        dec, cal = model_yaml["declarations"], model_yaml["calibration"]

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

        context = build_symbolic_context(var_ordering + par_ordering + shk_ordering + other_para)

        if "model" in model_yaml["equations"]:
            raw_equations = model_yaml["equations"]["model"]
        else:
            raw_equations = model_yaml["equations"]

        logger.debug(f"Processing {len(raw_equations)} model equations")
        equations = construct_equation_list(raw_equations, context)

        # Validate the leads and lags in the model
        logger.info("Validating model leads and lags")
        # Get the model-defined max_lead and max_lag from class or parameters
        max_lead = dec.get("max_lead", getattr(cls, "max_lead", 1))
        max_lag = dec.get("max_lag", getattr(cls, "max_lag", 1))
        
        # Use the validation module to check for excessive leads/lags
        validation_errors = validate_dsge_leads_lags(
            equations,
            var_ordering,
            max_lead=max_lead,
            max_lag=max_lag
        )
        
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Validation error: {error}")
            raise ValueError(
                "DSGE model validation failed. The following errors were found:\n" + 
                "\n".join(validation_errors)
            )
        
        # Check for general model consistency issues (only warnings)
        consistency_warnings = validate_model_consistency({
            'equations': equations, 
            'variables': var_ordering
        })
        
        for warning in consistency_warnings:
            logger.warning(f"Model consistency warning: {warning}")

        # ------------------------------------------------------------
        # Figure out max leads and lags
        # ------------------------------------------------------------
        logger.debug("Computing maximum leads and lags for shocks")
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
                # Pure symbol replacement: `xreplace` is much faster than `subs` for large models.
                equations = [eq.xreplace(subs_dict) for eq in equations]

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

            # leads
            for i in np.arange(2, abs(max_lead_endo[v]) + 1):
                # For lead i we add a (i-1)-step auxiliary variable, chaining one-period ahead.
                # Example: v(+2) becomes v_LEAD1(+1) with v_LEAD1 = v(+1).
                var_f = Variable(v.name + "_LEAD" + str(i - 1))

                if i == 2:
                    var_f_1 = Variable(v.name, date=1)
                else:
                    var_f_1 = Variable(v.name + "_LEAD" + str(i - 2), date=1)

                subs_dict[Variable(v.name, date=i)] = var_f(1)
                var_ordering.append(var_f)
                equations.append(Equation(var_f, var_f_1))

        # Pure symbol replacement: `xreplace` is much faster than `subs` for large models.
        equations = [eq.xreplace(subs_dict) for eq in equations]
        if "covariance" in cal:
            QQ = from_dict_to_mat(cal["covariance"], shk_ordering, context)
        else:
            logger.warning("No covariance matrix provided. Assuming identity matrix.")
            QQ = sympy.eye(len(shk_ordering))

        #------------------------------------------------------------------
        # observation equation
        #------------------------------------------------------------------
        context["sum"] = np.sum
        context["range"] = range
        for obs in obs_equations.items():
            # Parse observable equations safely
            obs_equations[obs[0]] = parse_expression(obs[1], context)

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
            "estimation": model_yaml['estimation'],
            "__data__": model_yaml,
            "name": dec["name"],
            "observables": observables,
            "obs_equations": obs_equations,
            "max_lead": max_lead,
            "max_lag": max_lag,
        }

        logger.info(f"DSGE model '{dec['name']}' creation complete with {len(var_ordering)} variables and {len(equations)} equations")
        model = cls(**model_dict)
        return model
