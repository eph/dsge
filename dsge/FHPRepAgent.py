#!/usr/bin/env python3
"""
FHP Representative Agent DSGE model implementation.

This module provides classes for working with FHP (Full Information Home Production)
Representative Agent DSGE models, which include cycle, trend, and value function components.
"""

import numpy as np
import sympy
import re
from typing import Dict, Optional, Any, Tuple

from sympy import sympify
from sympy.utilities.lambdify import lambdify

from .symbols import (Variable,
                      Shock,
                      Parameter)


from .Prior import construct_prior
from .data import read_data_file
from .StateSpaceModel import LinearDSGEModel
from .parsing_tools import from_dict_to_mat, construct_equation_list, parse_expression, build_symbolic_context
from .validation import check_for_future_shocks
from .logging_config import get_logger

from sympy.printing import fcode
from sympy.printing.fortran import FCodePrinter

from .Base import Base

# Get module logger
logger = get_logger("fhp")

# --------------------------------------------------------------------
# Horizon (k) parsing helpers
# --------------------------------------------------------------------

def _parse_k_spec(k_decl: Any) -> Dict[str, Any]:
    """Parse declarations.k into a normalized spec.

    Supported forms:
      - int: scalar horizon (backward compatible)
      - dict: {"default": int, "by_lhs": {lhs_name: int}}

    Returns a dict with keys: default, by_lhs, k_max.
    """
    if isinstance(k_decl, (int, np.integer)):
        k_default = int(k_decl)
        if k_default < 0:
            raise ValueError(f"declarations.k must be >= 0, got {k_default}")
        return {"default": k_default, "by_lhs": {}, "k_max": k_default}

    if isinstance(k_decl, dict):
        allowed_keys = {"default", "by_lhs", "k_max"}
        extra_keys = set(k_decl.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(
                "declarations.k dict supports only keys "
                f"{sorted(allowed_keys)}, got extra keys {sorted(extra_keys)}"
            )

        if "default" not in k_decl:
            raise ValueError("declarations.k dict must include a 'default' integer horizon")

        k_default = int(k_decl["default"])
        if k_default < 0:
            raise ValueError(f"declarations.k.default must be >= 0, got {k_default}")

        by_lhs_raw = k_decl.get("by_lhs", {}) or {}
        if not isinstance(by_lhs_raw, dict):
            raise ValueError("declarations.k.by_lhs must be a dict of {lhs_name: int}")

        k_by_lhs: Dict[str, int] = {}
        for lhs_name, k_val in by_lhs_raw.items():
            k_int = int(k_val)
            if k_int < 0:
                raise ValueError(
                    "declarations.k.by_lhs horizons must be >= 0, "
                    f"got {lhs_name!r}: {k_int}"
                )
            k_by_lhs[str(lhs_name)] = k_int

        k_max = max([k_default] + (list(k_by_lhs.values()) if k_by_lhs else []))
        return {"default": k_default, "by_lhs": k_by_lhs, "k_max": k_max}

    raise ValueError(
        "declarations.k must be an integer, or a dict like "
        "{default: <int>, by_lhs: {<lhs_name>: <int>, ...}}"
    )


def _align_terminal_equations_by_lhs_name(*, plan_dyn, term_dyn, block: str):
    """Align terminal dynamic equations to plan dynamic equations by LHS variable name.

    This is required for row-wise mixing of plan vs terminal equations.
    """
    def _lhs_name(eq, where: str) -> str:
        lhs = eq.lhs
        if not isinstance(lhs, Variable):
            raise ValueError(f"{where} equation LHS must be a Variable, got {type(lhs).__name__}: {lhs}")
        return lhs.name

    plan_names = [_lhs_name(eq, f"{block}.plan") for eq in plan_dyn]
    if len(set(plan_names)) != len(plan_names):
        dupes = sorted({n for n in plan_names if plan_names.count(n) > 1})
        raise ValueError(f"Duplicate LHS variable(s) in {block}.plan equations: {dupes}")

    term_names = [_lhs_name(eq, f"{block}.terminal") for eq in term_dyn]
    if len(set(term_names)) != len(term_names):
        dupes = sorted({n for n in term_names if term_names.count(n) > 1})
        raise ValueError(f"Duplicate LHS variable(s) in {block}.terminal equations: {dupes}")

    term_map = {name: eq for name, eq in zip(term_names, term_dyn)}

    missing = [name for name in plan_names if name not in term_map]
    if missing:
        raise ValueError(
            f"Missing {block}.terminal equation(s) for plan LHS variable(s): {missing}"
        )

    return [term_map[name] for name in plan_names]


def _svd_rank(mat: np.ndarray, *, tol: float) -> int:
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return 0
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return 0
    s0 = float(s[0])
    if not np.isfinite(s0) or s0 == 0.0:
        return 0
    cutoff = float(tol) * s0
    return int(np.sum(s > cutoff))


def _controllable_subspace_basis(
    TT: np.ndarray,
    RR: np.ndarray,
    *,
    tol: float,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Orthonormal basis Q for the reachable (controllable) subspace of (TT, RR).

    Builds a controllability matrix [RR, TT RR, ..., TT^{L-1} RR] and computes a
    rank-revealing SVD basis.
    """
    TT = np.asarray(TT, dtype=float)
    RR = np.asarray(RR, dtype=float)
    n = int(TT.shape[0])
    if RR.size == 0:
        return np.zeros((n, 0), dtype=float)

    L = int(n if max_steps is None else max(1, min(int(max_steps), n)))
    blocks = []
    M = RR
    for _ in range(L):
        blocks.append(M)
        M = TT @ M
    C = np.concatenate(blocks, axis=1)
    u, s, _ = np.linalg.svd(C, full_matrices=False)
    if s.size == 0:
        return np.zeros((n, 0), dtype=float)
    s0 = float(s[0])
    if not np.isfinite(s0) or s0 == 0.0:
        return np.zeros((n, 0), dtype=float)
    r = int(np.sum(s > float(tol) * s0))
    return np.ascontiguousarray(u[:, :r])


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_identifier(name: str) -> bool:
    return bool(_IDENT_RE.match(str(name)))


def _lagged_endogenous_variable_names(equations: Dict[str, Any], variables) -> set[str]:
    """Return endogenous variable names that appear with a negative lag anywhere."""
    endog_names = {v.name for v in variables}
    lagged: set[str] = set()

    def _scan(eq_list):
        for eq in eq_list:
            expr = eq.set_eq_zero
            for v in expr.atoms(Variable):
                if v.lag < 0 and v.name in endog_names:
                    lagged.add(v.name)

    _scan(equations.get("static", []))
    for blk in ("cycle", "trend"):
        if blk in equations:
            _scan(equations[blk].get("terminal", []))
            _scan(equations[blk].get("plan", []))
    if "value" in equations:
        _scan(equations["value"].get("function", []))
        _scan(equations["value"].get("update", []))

    return lagged


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
        k_cycle_row=None,
        k_trend_row=None,
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
        self.k = int(k)
        self.k_cycle_row = None if k_cycle_row is None else np.asarray(k_cycle_row, dtype=int)
        self.k_trend_row = None if k_trend_row is None else np.asarray(k_trend_row, dtype=int)

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

       nvar = A_cycle.shape[0]
       if self.k_cycle_row is None:
           k_cycle_row = np.full((nvar,), self.k, dtype=int)
       else:
           k_cycle_row = np.asarray(self.k_cycle_row, dtype=int)
           if k_cycle_row.shape != (nvar,):
               raise ValueError(
                   f"k_cycle_row must have shape ({nvar},), got {k_cycle_row.shape}"
               )

       if self.k_trend_row is None:
           k_trend_row = np.full((nvar,), self.k, dtype=int)
       else:
           k_trend_row = np.asarray(self.k_trend_row, dtype=int)
           if k_trend_row.shape != (nvar,):
               raise ValueError(
                   f"k_trend_row must have shape ({nvar},), got {k_trend_row.shape}"
               )

       for m in range(1, self.k + 1):
           # Cycle: mix plan vs terminal rows based on m <= k_cycle_row[i]
           alphaC_eff = alpha0_cycle.copy()
           alphaF_eff = np.zeros_like(alphaF_cycle)
           alphaB_eff = alpha1_cycle.copy()
           betaS_eff = beta0_cycle.copy()

           plan_rows = m <= k_cycle_row
           alphaC_eff[plan_rows, :] = alphaC_cycle[plan_rows, :]
           alphaF_eff[plan_rows, :] = alphaF_cycle[plan_rows, :]
           alphaB_eff[plan_rows, :] = alphaB_cycle[plan_rows, :]
           betaS_eff[plan_rows, :] = betaS_cycle[plan_rows, :]

           inv_cycle = np.linalg.inv(alphaC_eff - alphaF_eff @ A_cycle)
           A_cycle_new = inv_cycle @ alphaB_eff
           B_cycle_new = inv_cycle @ (alphaF_eff @ B_cycle @ P + betaS_eff)

           # Trend: mix plan vs terminal rows; terminal contributes value loading (betaV_trend)
           alphaC_eff = alpha0_trend.copy()
           alphaF_eff = np.zeros_like(alphaF_trend)
           alphaB_eff = alpha1_trend.copy()
           betaV_eff = betaV_trend.copy()

           plan_rows = m <= k_trend_row
           alphaC_eff[plan_rows, :] = alphaC_trend[plan_rows, :]
           alphaF_eff[plan_rows, :] = alphaF_trend[plan_rows, :]
           alphaB_eff[plan_rows, :] = alphaB_trend[plan_rows, :]
           betaV_eff[plan_rows, :] = 0.0

           inv_trend = np.linalg.inv(alphaC_eff - alphaF_eff @ A_trend)
           A_trend_new = inv_trend @ alphaB_eff
           B_trend_new = inv_trend @ (alphaF_eff @ B_trend + betaV_eff)

           A_cycle = A_cycle_new
           B_cycle = B_cycle_new
           A_trend = A_trend_new
           B_trend = B_trend_new

           if self.expectations > 0:
               A_cycle_history[m] = A_cycle.copy()
               B_cycle_history[m] = B_cycle.copy()

               A_trend_history[m] = A_trend.copy()
               B_trend_history[m] = B_trend.copy()

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
        k_in = self['k'] if k is None else k
        expectations = self['expectations'] if expectations is None else expectations
        cmodel = self.compile_model(k=k_in, expectations=expectations)
     
        npara = len(self['parameters'])
        para = sympy.IndexedBase("para", shape=(npara + 1,))
     
     
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
     
        context = build_symbolic_context(list(self['parameters']) + list(self['auxiliary_parameters'].keys()))
     
        to_replace = {}
        for op, expr in self["auxiliary_parameters"].items():
            to_replace[op] = parse_expression(str(expr), context)
     
        to_replace = list(to_replace.items())
     
        from itertools import permutations
     
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

        # Generate Fortran code for data matrix
        # Note: fortress expects yy as (nobs, T) not (T, nobs)
        import numpy as np
        import pandas as pd

        # Generate custom prior code using the same approach as regular DSGE models
        from .translate import generate_custom_prior_fortran, generate_hardcoded_data_fortran


        data_fortran = generate_hardcoded_data_fortran(cmodel.yy)

        custom_prior_code = generate_custom_prior_fortran(cmodel.prior) if cmodel.prior is not None else ""

        # get templates/fhp.f90 via importlib.resources (zip-safe)
        from importlib.resources import files
        from .template_utils import render_template, build_fhp_placeholders
        template_res = files("dsge") / "templates" / "fhp.f90"
        with template_res.open("r", encoding="utf-8") as f:
            fortran_template = f.read()

        # Safely render template by explicit placeholder replacement
        placeholders = build_fhp_placeholders(
            nobs=cmodel.yy.shape[1],
            T=cmodel.yy.shape[0],
            nvar=len(self['variables']),
            nval=len(self['values']),
            nshock=len(self['shocks']),
            npara=len(self['parameters']),
            neps=len(self['innovations']),
            k=cmodel.k,
            k_cycle_row=cmodel.k_cycle_row,
            k_trend_row=cmodel.k_trend_row,
            t0=t0,
            system=sims_mat,
            data=data_fortran,
            custom_prior_code=custom_prior_code,
        )

        return render_template(fortran_template, placeholders, strict=True)



    @classmethod
    def read(cls, model_yaml: Dict[str, Any], k: Optional[int] = None) -> "FHPRepAgent":
        """
        Read a model specification from a YAML dictionary and create an FHPRepAgent model.
        
        Args:
            model_yaml: Dictionary containing the model specification
            k: Optional parameter to override the k value in the YAML
            
        Returns:
            An initialized FHPRepAgent model
            
        Raises:
            ValueError: If the model specification is invalid
            AssertionError: If equation counts don't match variable counts
        """
        logger.info("Reading FHP model from YAML specification")
        
        # Process declarations
        dec = model_yaml['declarations']
        variables = [Variable(v) for v in dec['variables']]
        values = [Variable(v) for v in dec['values']]
        value_updates = [Variable(v) for v in dec['value_updates']]
        shocks = [Variable(v) for v in dec['shocks']]
        innovations = [Shock(v) for v in dec['innovations']]
        parameters = [Parameter(v) for v in dec['parameters']]
        expectations = dec.get('expectations', 0)  # Use get with default for cleaner code
        
        logger.debug(f"Model has {len(variables)} variables, {len(shocks)} shocks, and {len(parameters)} parameters")

        # Process auxiliary parameters
        if "auxiliary_parameters" in dec:
            logger.debug(f"Processing {len(dec['auxiliary_parameters'])} auxiliary parameters")
            other_para = [Parameter(v) for v in dec["auxiliary_parameters"]]
            
            # Create a context for parsing
            param_context = {str(x): x for x in parameters + other_para}
            
            try:
                other_para = {
                    op: sympify(
                        model_yaml['calibration']["auxiliary_parameters"][op.name],
                        param_context
                    ) for op in other_para
                }
            except KeyError as e:
                logger.error(f"Missing auxiliary parameter definition: {e}")
                raise ValueError(f"Auxiliary parameter {e} is declared but not defined in calibration")
        else:
            other_para = {}

        # Process measurement errors
        if "measurement_errors" in dec:
            measurement_errors = [Shock(v) for v in dec["measurement_errors"]]
            logger.debug(f"Model includes {len(measurement_errors)} measurement errors")
        else:
            measurement_errors = None

        # Create parsing context with all symbols
        context = build_symbolic_context(variables + values + value_updates + shocks + innovations + parameters + list(other_para.keys()))

        # Process observables
        if "observables" in dec:
            logger.debug(f"Processing {len(dec['observables'])} observables")
            observables = [Variable(v) for v in dec["observables"]]
            
            # Create observables equations
            try:
                obs_equations = {
                    o: parse_expression(model_yaml["model"]["observables"][str(o)], context)
                    for o in observables
                }
            except KeyError as e:
                logger.error(f"Missing observable equation: {e}")
                raise ValueError(f"Observable {e} is declared but has no equation in model.observables")
        else:
            # Default: use variables as observables
            observables = [Variable(v) for v in dec["variables"]]
            obs_equations = {v: v for v in observables}
            logger.debug("No observables specified, using variables as observables")

        # Set up the equations
        yaml_eq = model_yaml['model']
        equations = {}
        
        # Process static equations (optional section)
        if 'static' in yaml_eq:
            logger.debug(f"Processing {len(yaml_eq['static'])} static equations")
            equations['static'] = construct_equation_list(yaml_eq['static'], context)
        else:
            equations['static'] = []
            logger.debug("No static equations specified")

        # Process cycle equations (required sections)
        logger.debug(f"Processing cycle equations (terminal: {len(yaml_eq['cycle']['terminal'])}, plan: {len(yaml_eq['cycle']['plan'])})")
        equations['cycle'] = {
            'terminal': construct_equation_list(yaml_eq['cycle']['terminal'], context),
            'plan': construct_equation_list(yaml_eq['cycle']['plan'], context)
        }

        # Verify equation counts match variable counts
        try:
            assert len(equations['cycle']['terminal']) + len(equations['static']) == len(variables)
            assert len(equations['cycle']['plan']) + len(equations['static']) == len(variables)
        except AssertionError:
            var_count = len(variables)
            static_count = len(equations['static']) 
            terminal_count = len(equations['cycle']['terminal'])
            plan_count = len(equations['cycle']['plan'])
            logger.error(
                f"Equation count mismatch: variables={var_count}, "
                f"static={static_count}, cycle.terminal={terminal_count}, cycle.plan={plan_count}"
            )
            raise AssertionError(
                f"FHP model requires exactly {var_count} equations, but found "
                f"{static_count + terminal_count} terminal equations and {static_count + plan_count} plan equations"
            )

        # Process trend equations (required sections)
        logger.debug(f"Processing trend equations (terminal: {len(yaml_eq['trend']['terminal'])}, plan: {len(yaml_eq['trend']['plan'])})")
        equations['trend'] = {
            'terminal': construct_equation_list(yaml_eq['trend']['terminal'], context),
            'plan': construct_equation_list(yaml_eq['trend']['plan'], context)
        }

        # Verify equation counts match variable counts for trend equations
        try:
            assert len(equations['trend']['terminal']) + len(equations['static']) == len(variables)
            assert len(equations['trend']['plan']) + len(equations['static']) == len(variables)
        except AssertionError:
            var_count = len(variables)
            static_count = len(equations['static']) 
            terminal_count = len(equations['trend']['terminal'])
            plan_count = len(equations['trend']['plan'])
            logger.error(
                f"Trend equation count mismatch: variables={var_count}, "
                f"static={static_count}, trend.terminal={terminal_count}, trend.plan={plan_count}"
            )
            raise AssertionError(
                f"FHP model requires exactly {var_count} equations, but found "
                f"{static_count + terminal_count} terminal equations and {static_count + plan_count} plan equations"
            )

        # Process value function equations
        logger.debug(f"Processing value equations (function: {len(yaml_eq['value']['function'])}, update: {len(yaml_eq['value']['update'])})")
        equations['value'] = {
            'function': construct_equation_list(yaml_eq['value']['function'], context),
            'update': construct_equation_list(yaml_eq['value']['update'], context)
        }

        # Process shock equations
        logger.debug(f"Processing {len(yaml_eq['shocks'])} shock equations")
        equations['shocks'] = construct_equation_list(yaml_eq['shocks'], context)
        
        # Helper function to get original equation text for error messages
        def get_original_equation(eq_idx: int, equation_type: str) -> str:
            """Get the original equation text from the YAML for better error messages."""
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
        
        # Validate that no future shocks are used in any equations
        logger.info("Validating model: checking for future shocks")
        
        # Check static equations
        if equations['static']:
            check_for_future_shocks(
                equations['static'], 
                shocks, 
                'static',
                get_original_equation
            )
            
        # Check cycle equations
        for section in ['terminal', 'plan']:
            check_for_future_shocks(
                equations['cycle'][section], 
                shocks, 
                f'cycle/{section}',
                get_original_equation
            )
            
        # Check trend equations
        for section in ['terminal', 'plan']:
            check_for_future_shocks(
                equations['trend'][section], 
                shocks, 
                f'trend/{section}',
                get_original_equation
            )
            
        # Check value function equations
        for section in ['function', 'update']:
            check_for_future_shocks(
                equations['value'][section], 
                shocks, 
                f'value/{section}',
                get_original_equation
            )
            
        logger.info("Model validation passed: no future shocks found")

        # Process covariance matrix
        if 'covariance' in model_yaml['calibration']:
            logger.debug("Processing covariance matrix from calibration")
            QQ = from_dict_to_mat(model_yaml['calibration']['covariance'], innovations, context)
        else:
            logger.warning('No covariance matrix provided. Assuming identity matrix.')
            QQ = sympy.eye(len(innovations))

        # Process measurement errors
        me_dict = {}
        if 'measurement_errors' in model_yaml['calibration']:
            me_dict = model_yaml['calibration']['measurement_errors']
            logger.debug("Processing measurement error specifications from calibration")

        # Create measurement error covariance matrix
        if measurement_errors is not None:
            logger.debug(f"Creating measurement error covariance matrix of size {len(measurement_errors)}x{len(measurement_errors)}")
            HH = from_dict_to_mat(me_dict, measurement_errors, context)
        else:
            logger.debug(f"Creating measurement error covariance matrix of size {len(observables)}x{len(observables)}")
            HH = from_dict_to_mat(me_dict, observables, context)

        # Create the model dictionary with all components
        k_spec = _parse_k_spec(k if k is not None else dec["k"])

        if "horizon_choice" in dec and "stopping_rule" in dec:
            raise ValueError("Use only one of declarations.horizon_choice or declarations.stopping_rule.")
        horizon_choice = dec.get("horizon_choice", None)
        if horizon_choice is None:
            horizon_choice = dec.get("stopping_rule", None)

        model_dict = {
            'variables': variables,
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
            # Backward compatible scalar k for any callers that expect an integer.
            'k': k_spec["k_max"],
            # Rich horizon spec for equation-row horizons.
            'k_spec': k_spec,
            # Optional endogenous horizon-choice configuration.
            'horizon_choice': horizon_choice,
        }

        logger.info(f"FHP model creation complete: {len(variables)} variables, {len(parameters)} parameters")
        return cls(**model_dict)

    def compile_model(self, k=None,expectations=None):

        k_spec = _parse_k_spec(self.get("k_spec", self["k"]) if k is None else k)
        k = k_spec["k_max"]
        expectations = self['expectations'] if expectations is None else expectations

        nv = len(self['variables'])
        ns = len(self['shocks'])
        nval = len(self['values'])
        v = self['variables']

        static_eqs = self["equations"]["static"]

        cycle_plan_dyn = self["equations"]["cycle"]["plan"]
        cycle_term_dyn = self["equations"]["cycle"]["terminal"]
        cycle_term_dyn_aligned = _align_terminal_equations_by_lhs_name(
            plan_dyn=cycle_plan_dyn, term_dyn=cycle_term_dyn, block="cycle"
        )
        cycle_term_eqs = cycle_term_dyn_aligned + static_eqs
        cycle_plan_eqs = cycle_plan_dyn + static_eqs

        self.alpha0_cycle = sympy.Matrix(nv, nv , lambda i, j: cycle_term_eqs[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_cycle = sympy.Matrix(nv, nv , lambda i, j: -cycle_term_eqs[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.beta0_cycle = sympy.Matrix(nv, ns , lambda i, j: -cycle_term_eqs[i].set_eq_zero.diff(self['shocks'][j]))

        self.alphaC_cycle = sympy.Matrix(nv, nv, lambda i, j: cycle_plan_eqs[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_plan_eqs[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_cycle = sympy.Matrix(nv, nv, lambda i, j: -cycle_plan_eqs[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaS_cycle = sympy.Matrix(nv, ns, lambda i ,j: -cycle_plan_eqs[i].set_eq_zero.diff(self['shocks'][j]))

        trend_plan_dyn = self["equations"]["trend"]["plan"]
        trend_term_dyn = self["equations"]["trend"]["terminal"]
        trend_term_dyn_aligned = _align_terminal_equations_by_lhs_name(
            plan_dyn=trend_plan_dyn, term_dyn=trend_term_dyn, block="trend"
        )
        trend_term_eqs = trend_term_dyn_aligned + static_eqs
        trend_plan_eqs = trend_plan_dyn + static_eqs

        all_var_names = {var.name for var in self["variables"]}
        unknown_overrides = sorted(set(k_spec["by_lhs"].keys()) - all_var_names)
        if unknown_overrides:
            raise ValueError(
                "declarations.k.by_lhs contains unknown variable name(s): "
                f"{unknown_overrides}. Expected one of {sorted(all_var_names)}."
            )

        def _build_row_horizons(eq_list, block: str) -> np.ndarray:
            if len(eq_list) != nv:
                raise ValueError(
                    f"Internal error: {block} equation list has {len(eq_list)} rows, expected {nv}"
                )
            out = np.zeros((nv,), dtype=int)
            for i, eq in enumerate(eq_list):
                lhs = eq.lhs
                if not isinstance(lhs, Variable):
                    raise ValueError(
                        f"{block} equation row {i} LHS must be a Variable, got {type(lhs).__name__}: {lhs}"
                    )
                out[i] = k_spec["by_lhs"].get(lhs.name, k_spec["default"])
            return out

        k_cycle_row = _build_row_horizons(cycle_plan_eqs, "cycle.plan")
        k_trend_row = _build_row_horizons(trend_plan_eqs, "trend.plan")

        self.alpha0_trend = sympy.Matrix(nv, nv , lambda i, j: trend_term_eqs[i].set_eq_zero.diff(self['variables'][j]))
        self.alpha1_trend = sympy.Matrix(nv, nv , lambda i, j: -trend_term_eqs[i].set_eq_zero.diff(self['variables'][j](-1)))
        self.betaV_trend = sympy.Matrix(nv, nval , lambda i, j: -trend_term_eqs[i].set_eq_zero.diff(self['values'][j]))

        self.alphaC_trend = sympy.Matrix(nv, nv, lambda i, j: trend_plan_eqs[i].set_eq_zero.diff(self['variables'][j]))
        self.alphaF_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_plan_eqs[i].set_eq_zero.diff(self['variables'][j](+1)))
        self.alphaB_trend = sympy.Matrix(nv, nv, lambda i, j: -trend_plan_eqs[i].set_eq_zero.diff(self['variables'][j](-1)))

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
                                          HH, k,
                                          k_cycle_row=k_cycle_row,
                                          k_trend_row=k_trend_row,
                                          t0=0, expectations=expectations,
                                          shock_names=shock_names,
                                          state_names=state_names,
                                          obs_names=obs_names,
                                          prior=prior,
                                          parameter_names=parameter_names)
        return linmod

    def compile_endogenous_horizon_model(
        self,
        *,
        expectations: Optional[int] = None,
        basis_tol: float = 1e-10,
        basis_max_steps: Optional[int] = None,
    ):
        """
        Compile an endogenous-horizon switching model from an FHP YAML with
        declarations.stopping_rule (or declarations.horizon_choice).

        Returns an EndogenousHorizonSwitchingModel whose state is a reduced set of
        predetermined variables (selected components of the full FHP state). The
        observation matrix becomes regime-dependent via reconstruction of the full
        state on the reachable subspace.
        """
        hc = self.get("horizon_choice", None)
        if hc is None:
            raise ValueError("Missing declarations.stopping_rule (or declarations.horizon_choice) in FHP YAML.")

        expectations = self["expectations"] if expectations is None else int(expectations)

        # Base horizon spec (can include fixed by_lhs overrides).
        k_base_spec = _parse_k_spec(self.get("k_spec", self["k"]))
        all_var_names = {v.name for v in self["variables"]}
        unknown_base = sorted(set(k_base_spec["by_lhs"].keys()) - all_var_names)
        if unknown_base:
            raise ValueError(
                "declarations.k.by_lhs contains unknown variable name(s): "
                f"{unknown_base}. Expected one of {sorted(all_var_names)}."
            )

        hc_components = hc.get("components", {}) or {}
        if not isinstance(hc_components, dict) or not hc_components:
            raise ValueError("declarations.stopping_rule.components (or declarations.horizon_choice.components) must be a non-empty dict.")

        components = [str(c) for c in hc_components.keys()]
        if len(set(components)) != len(components):
            raise ValueError(
                "Duplicate components in declarations.stopping_rule.components (or declarations.horizon_choice.components): "
                f"{components}"
            )

        selection_order = hc.get("selection_order", None)
        if selection_order is None:
            selection_order = components
        else:
            selection_order = [str(c) for c in selection_order]
            if set(selection_order) != set(components):
                raise ValueError(
                    "declarations.stopping_rule.selection_order (or declarations.horizon_choice.selection_order) "
                    "must be a permutation of components."
                )

        # Validate LHS assignments and build per-component configs.
        cycle_plan_dyn = self["equations"]["cycle"]["plan"]
        trend_plan_dyn = self["equations"]["trend"]["plan"]
        cycle_plan_lhs = [eq.lhs.name for eq in cycle_plan_dyn]
        trend_plan_lhs = [eq.lhs.name for eq in trend_plan_dyn]

        lhs_owner: Dict[str, str] = {}
        assign_lhs_by_comp: Dict[str, list[str]] = {}
        k_max_by_comp: Dict[str, int] = {}

        # Parse cost/lambda/policy expressions (evaluated at calibration p0 for now).
        # We keep parsing strict to avoid silent typos.
        p0 = np.asarray(self.p0(), dtype=float)
        parameter_names = [str(x) for x in self["parameters"]]
        param_syms = {name: sympy.Symbol(name) for name in parameter_names}
        param_ctx = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt, "Abs": sympy.Abs, **param_syms}
        allowed_param_syms = set(param_syms.values())

        def _parse_param_expr(expr: Any, *, where: str) -> sympy.Expr:
            s = str(expr) if not isinstance(expr, str) else expr
            try:
                out = sympy.sympify(s, locals=param_ctx)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"While parsing {where} expression {s!r}: {e}") from e
            unknown = [v for v in out.free_symbols if v not in allowed_param_syms]
            if unknown:
                raise ValueError(f"Unknown symbol(s) in {where} expression {s!r}: {[str(u) for u in unknown]}")
            return out

        a_by_comp: Dict[str, float] = {}
        lam_by_comp: Dict[str, float] = {}
        policy_expr_by_comp: Dict[str, sympy.Expr] = {}

        for comp in components:
            cfg = hc_components[comp]
            try:
                k_max_by_comp[comp] = int(cfg["k_max"])
            except Exception as e:
                raise ValueError(f"horizon_choice.components.{comp}.k_max must be an integer.") from e
            if k_max_by_comp[comp] < 0:
                raise ValueError(
                    f"horizon_choice.components.{comp}.k_max must be >= 0, got {k_max_by_comp[comp]}"
                )

            lhs_list = cfg.get("assign_lhs", None)
            if not isinstance(lhs_list, (list, tuple)) or not lhs_list:
                raise ValueError(f"horizon_choice.components.{comp}.assign_lhs must be a non-empty list.")
            lhs_names = [str(x) for x in lhs_list]
            unknown_assign = sorted(set(lhs_names) - all_var_names)
            if unknown_assign:
                raise ValueError(
                    f"horizon_choice.components.{comp}.assign_lhs has unknown variable(s): {unknown_assign}."
                )
            missing_cycle = sorted(set(lhs_names) - set(cycle_plan_lhs))
            missing_trend = sorted(set(lhs_names) - set(trend_plan_lhs))
            if missing_cycle or missing_trend:
                raise ValueError(
                    f"horizon_choice.components.{comp}.assign_lhs must appear in both cycle.plan and trend.plan. "
                    f"Missing in cycle.plan: {missing_cycle}; missing in trend.plan: {missing_trend}."
                )

            for nm in lhs_names:
                prev = lhs_owner.get(nm)
                if prev is not None and prev != comp:
                    raise ValueError(
                        f"LHS variable {nm!r} is assigned to multiple components: {prev!r}, {comp!r}."
                    )
                lhs_owner[nm] = comp
            assign_lhs_by_comp[comp] = lhs_names

            if "cost" not in cfg or not isinstance(cfg["cost"], dict) or "a" not in cfg["cost"]:
                raise ValueError(f"horizon_choice.components.{comp}.cost.a is required.")
            a_expr = _parse_param_expr(cfg["cost"]["a"], where=f"horizon_choice.components.{comp}.cost.a")
            a_val = float(sympy.lambdify(list(param_syms.values()), a_expr, modules="numpy")(*p0.tolist()))
            a_by_comp[comp] = a_val

            lam_expr = _parse_param_expr(cfg["lambda"], where=f"horizon_choice.components.{comp}.lambda")
            lam_val = float(sympy.lambdify(list(param_syms.values()), lam_expr, modules="numpy")(*p0.tolist()))
            lam_by_comp[comp] = lam_val

            pol = cfg.get("policy_object", None)
            if not isinstance(pol, str) or not pol.strip():
                raise ValueError(f"horizon_choice.components.{comp}.policy_object must be a non-empty string.")
            # Parsed later once reduced state/observable symbols are known.
            policy_expr_by_comp[comp] = sympy.sympify(pol, locals={})

        global_k_max = max([k_base_spec["k_max"]] + [k_max_by_comp[c] for c in components])

        # Compile a kernel FHP linear model once at a global maximum horizon.
        kernel = self.compile_model(k=global_k_max, expectations=expectations)

        # Select a reduced switching state:
        # - lagged endogenous variables (total) (predetermined)
        # - values
        # - shock states
        lagged_names = _lagged_endogenous_variable_names(self["equations"], self["variables"])
        base_state_names = [nm for nm in kernel.state_names if nm in lagged_names]
        base_state_names += [str(v) for v in self["values"]]
        base_state_names += [str(s) for s in self["shocks"]]

        # Avoid selecting controlled variables if possible (so observables can depend on regime).
        controlled = set()
        for lhs in lhs_owner.keys():
            controlled.update({lhs, f"{lhs}_cycle", f"{lhs}_trend"})

        state_index = {name: i for i, name in enumerate(kernel.state_names)}
        missing = [nm for nm in base_state_names if nm not in state_index]
        if missing:
            raise ValueError(f"Internal error: reduced-state name(s) missing from kernel.state_names: {missing}")

        sel_idx = []
        seen = set()
        for nm in base_state_names:
            i = state_index[nm]
            if i not in seen:
                sel_idx.append(i)
                seen.add(i)
        sel_idx.sort()

        # Build a reference regime (all components at k=0) for rank-checking.
        regime0 = tuple(0 for _ in components)

        static_eqs = self["equations"]["static"]
        cycle_plan_eqs = self["equations"]["cycle"]["plan"] + static_eqs
        trend_plan_eqs = self["equations"]["trend"]["plan"] + static_eqs
        nv = len(self["variables"])

        def _row_horizons_for_regime(regime: Tuple[int, ...]):
            by_lhs = dict(k_base_spec["by_lhs"])
            for comp, kval in zip(components, regime):
                for lhs in assign_lhs_by_comp[comp]:
                    by_lhs[lhs] = int(kval)
            default_k = int(k_base_spec["default"])

            def _build(eq_list, block: str) -> np.ndarray:
                if len(eq_list) != nv:
                    raise ValueError(f"Internal error: {block} has {len(eq_list)} rows, expected {nv}")
                out = np.zeros((nv,), dtype=int)
                for i, eq in enumerate(eq_list):
                    lhs = eq.lhs
                    if not isinstance(lhs, Variable):
                        raise ValueError(
                            f"Internal error: {block} row {i} LHS must be a Variable, got {type(lhs).__name__}: {lhs}"
                        )
                    out[i] = int(by_lhs.get(lhs.name, default_k))
                return out

            return _build(cycle_plan_eqs, "cycle.plan"), _build(trend_plan_eqs, "trend.plan")

        def _full_mats_at_regime(params_vec: np.ndarray, regime: Tuple[int, ...]):
            k_cycle_row, k_trend_row = _row_horizons_for_regime(regime)
            k_use = int(max(int(np.max(k_cycle_row)), int(np.max(k_trend_row))))
            lm = LinearDSGEforFHPRepAgent(
                kernel.yy,
                kernel.alpha0_cycle,
                kernel.alpha1_cycle,
                kernel.beta0_cycle,
                kernel.alphaC_cycle,
                kernel.alphaF_cycle,
                kernel.alphaB_cycle,
                kernel.betaS_cycle,
                kernel.alpha0_trend,
                kernel.alpha1_trend,
                kernel.betaV_trend,
                kernel.alphaC_trend,
                kernel.alphaF_trend,
                kernel.alphaB_trend,
                kernel.value_gammaC,
                kernel.value_gamma,
                kernel.value_Cx,
                kernel.value_Cs,
                kernel.P,
                kernel.R,
                kernel.QQ,
                kernel.DD,
                kernel.ZZ,
                kernel.HH,
                k_use,
                k_cycle_row=k_cycle_row,
                k_trend_row=k_trend_row,
                t0=kernel.t0,
                expectations=kernel.expectations,
                shock_names=kernel.shock_names,
                state_names=kernel.state_names,
                obs_names=kernel.obs_names,
                prior=kernel.prior,
                parameter_names=kernel.parameter_names,
            )
            return lm.system_matrices(params_vec)

        # Expand selection until it is injective on the reachable subspace.
        CC0, TT0, RR0, QQ0, DD0, ZZ0, HH0 = _full_mats_at_regime(p0, regime0)
        Q0 = _controllable_subspace_basis(TT0, RR0, tol=basis_tol, max_steps=basis_max_steps)
        r0 = int(Q0.shape[1])

        def _rank_on_sel(Q: np.ndarray, idx: list[int]) -> int:
            if Q.size == 0:
                return 0
            return _svd_rank(Q[np.asarray(idx, dtype=int), :], tol=basis_tol)

        if len(sel_idx) < r0:
            # start by allowing additional non-controlled identifiers from the full state.
            candidates = [
                i
                for i, nm in enumerate(kernel.state_names)
                if _is_identifier(nm) and (i not in seen) and (nm not in controlled)
            ]
            for i in candidates:
                sel_idx.append(i)
                seen.add(i)
                sel_idx.sort()
                if len(sel_idx) >= r0 and _rank_on_sel(Q0, sel_idx) == r0:
                    break

        if _rank_on_sel(Q0, sel_idx) != r0:
            # Last resort: allow adding controlled names too.
            candidates = [i for i, nm in enumerate(kernel.state_names) if _is_identifier(nm) and (i not in seen)]
            for i in candidates:
                sel_idx.append(i)
                seen.add(i)
                sel_idx.sort()
                if len(sel_idx) >= r0 and _rank_on_sel(Q0, sel_idx) == r0:
                    break

        if _rank_on_sel(Q0, sel_idx) != r0:
            raise ValueError(
                "Could not construct a reduced switching state that identifies the reachable subspace. "
                "Try simplifying observables or increasing the state selection manually (not yet supported)."
            )

        reduced_state_names = [kernel.state_names[i] for i in sel_idx]

        # Parse policy_object expressions now that state/observable symbols are known.
        obs_names = list(kernel.obs_names) if kernel.obs_names is not None else []
        if not obs_names:
            raise ValueError("Internal error: kernel has empty obs_names.")
        state_syms = {nm: sympy.Symbol(nm) for nm in reduced_state_names}
        obs_syms = {nm: sympy.Symbol(nm) for nm in obs_names}
        pol_ctx = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt, "Abs": sympy.Abs, **param_syms, **state_syms, **obs_syms}
        allowed_pol_syms = set(param_syms.values()) | set(state_syms.values()) | set(obs_syms.values())

        overlap_param_state = sorted(set(parameter_names) & set(reduced_state_names))
        if overlap_param_state:
            raise ValueError(
                "Parameter name(s) collide with switching-state name(s): "
                f"{overlap_param_state}. Rename one side to avoid ambiguity."
            )
        overlap_param_obs = sorted(set(parameter_names) & set(obs_names))
        if overlap_param_obs:
            raise ValueError(
                "Parameter name(s) collide with observable name(s): "
                f"{overlap_param_obs}. Rename one side to avoid ambiguity."
            )

        policy_funcs = {}
        # Build a de-duplicated lambdify argument list.
        #
        # This matters when observables overlap in name with state entries (e.g. when
        # you omit declarations.observables and FHP defaults to identity observables).
        pol_args_all = (
            [param_syms[n] for n in parameter_names]
            + [state_syms[nm] for nm in reduced_state_names]
            + [obs_syms[nm] for nm in obs_names]
        )
        pol_args = []
        seen = set()
        for s in pol_args_all:
            if s in seen:
                continue
            pol_args.append(s)
            seen.add(s)

        for comp in components:
            expr_raw = hc_components[comp]["policy_object"]
            expr = sympy.sympify(str(expr_raw), locals=pol_ctx)
            unknown = [v for v in expr.free_symbols if v not in allowed_pol_syms]
            if unknown:
                raise ValueError(
                    f"Unknown symbol(s) in horizon_choice.components.{comp}.policy_object: {[str(u) for u in unknown]}"
                )
            policy_funcs[comp] = lambdify(pol_args, expr, modules="numpy")

        # Build the switching model.
        from .endogenous_horizon_switching import EndogenousHorizonSwitchingModel

        model_ref: Dict[str, Any] = {}

        def solve_given_regime(params_vec: np.ndarray, regime: Tuple[int, ...]):
            CC, TT, RR, QQ, DD, ZZ, HH = _full_mats_at_regime(params_vec, tuple(int(x) for x in regime))

            Q = _controllable_subspace_basis(TT, RR, tol=basis_tol, max_steps=basis_max_steps)
            r = int(Q.shape[1])
            if r == 0:
                raise ValueError("No stochastic controllable directions; cannot form reduced switching state.")

            Qs = Q[np.asarray(sel_idx, dtype=int), :]
            if _svd_rank(Qs, tol=basis_tol) != r:
                raise ValueError(
                    "Reduced switching state is not injective on reachable subspace for this regime."
                )
            G = Q @ np.linalg.pinv(Qs, rcond=basis_tol)

            TT_red = TT[np.asarray(sel_idx, dtype=int), :] @ G
            RR_red = RR[np.asarray(sel_idx, dtype=int), :]
            ZZ_red = ZZ @ G
            DD_red = np.asarray(DD, dtype=float).reshape(-1)
            QQ = np.asarray(QQ, dtype=float)
            HH = np.asarray(HH, dtype=float)
            return TT_red, RR_red, ZZ_red, DD_red, QQ, HH

        def info_func(x_t: np.ndarray, t: int, chosen):
            x_t = np.asarray(x_t, dtype=float).reshape(-1)
            if x_t.shape != (len(reduced_state_names),):
                raise ValueError(
                    f"Switching state must have shape ({len(reduced_state_names)},), got {x_t.shape}."
                )
            info = {"x": x_t, "t": int(t), "chosen": dict(chosen)}
            for i, nm in enumerate(reduced_state_names):
                info[nm] = float(x_t[i])
            return info

        def policy_object(params_vec: np.ndarray, info_t, component: str, k: int, chosen_regime):
            m = model_ref.get("model")
            if m is None:  # pragma: no cover
                raise RuntimeError("Internal error: switching model not initialized.")
            x_t = np.asarray(info_t["x"], dtype=float).reshape(-1)
            k_by_comp = {c: int(chosen_regime.get(c, 0)) for c in components}
            k_by_comp[str(component)] = int(k)
            reg = tuple(int(k_by_comp[c]) for c in components)
            TT, RR, ZZ, DD, QQ, HH = m.get_mats(params_vec, reg)
            y_hat = ZZ @ x_t + np.asarray(DD, dtype=float).reshape(-1)

            # Map symbol -> numeric value; if a name overlaps between state and
            # observables, the observable value wins.
            val_by_sym = {}
            params_vec = np.asarray(params_vec, dtype=float).reshape(-1)
            for i, nm in enumerate(parameter_names):
                val_by_sym[param_syms[nm]] = float(params_vec[i])
            for i, nm in enumerate(reduced_state_names):
                val_by_sym.setdefault(state_syms[nm], float(x_t[i]))
            for i, nm in enumerate(obs_names):
                val_by_sym[obs_syms[nm]] = float(y_hat[i])

            args = [val_by_sym[s] for s in pol_args]
            return np.asarray(policy_funcs[str(component)](*args), dtype=float)

        out_model = EndogenousHorizonSwitchingModel(
            components=components,
            k_max=k_max_by_comp,
            cost_params={c: (a_by_comp[c], 0.0) for c in components},
            lam=lam_by_comp,
            solve_given_regime=solve_given_regime,
            policy_object=policy_object,
            info_func=info_func,
            selection_order=selection_order,
        )

        model_ref["model"] = out_model

        # Attach helpful metadata.
        setattr(out_model, "fhp_model", self)
        setattr(out_model, "state_names", list(reduced_state_names))
        setattr(out_model, "obs_names", list(obs_names))
        setattr(out_model, "shock_names", list(kernel.shock_names) if kernel.shock_names is not None else None)
        setattr(out_model, "parameter_names", list(parameter_names))
        setattr(out_model, "p0", p0)

        return out_model
