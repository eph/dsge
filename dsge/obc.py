"""
Occasionally Binding Constraints (OBC) support (skeleton).

Provides an OccBin-style wrapper model that holds per-regime linear DSGE models
and constraint metadata, plus placeholder IRF/simulation methods.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import sympy as sp

from .DSGE import DSGE


class OccBinModel:
    """Wrapper for an OBC model with two regimes and a single constraint.

    Attributes
    - regimes: dict[str, DSGE] parsed from YAML 'regimes'
    - constraint: dict with keys: name, when (str), binding_regime, normal_regime, horizon (opt)
    - observables, parameters, shocks: shared across regimes
    - context: parsing context (if needed later)
    """

    def __init__(self, *, regimes: Dict[str, DSGE], constraint: Dict[str, Any], yaml: Dict[str, Any]):
        self.regimes = regimes
        self.constraint = constraint
        self.yaml = yaml

        # Borrow shared attributes from one regime (we require consistency)
        any_regime = next(iter(regimes.values()))
        self.parameters = any_regime.parameters
        self.shocks = any_regime.shocks
        self.variables = any_regime.variables
        self.observables = any_regime["observables"]

        # Prepare convenience names
        self.normal_regime = constraint.get("normal_regime") or list(regimes.keys())[0]
        self.binding_regime = constraint.get("binding_regime") or self.normal_regime
        self.horizon_default = int(constraint.get("horizon", 40))

        # Parse the 'when' condition to a fast lambda
        self.when_expr = constraint.get("when", "")
        self._when_vars, self._when_lambda = self._compile_when(self.when_expr)

    def p0(self):
        return list(map(lambda x: self.yaml["calibration"]["parameters"][str(x)], self.parameters))

    def compile_regime(self, name: str):
        return self.regimes[name].compile_model()

    # Placeholder solver API — use normal regime IRF for now
    def irf_obc(self, p0, shock_name: Optional[str] = None, h: int = 20):
        # Build shocks path for a one-time 1-s.d. impulse
        normal = self.compile_regime(self.normal_regime)
        CC, TTn, RRn, QQn, DDn, ZZn, HHn = normal.system_matrices(p0)
        neps = RRn.shape[1]
        shock_names = normal.shock_names
        if shock_name is None:
            shock_idx = 0
        else:
            shock_idx = shock_names.index(shock_name)
        eps = np.zeros((h+1, neps))
        eps[0, shock_idx] = float(np.sqrt(QQn[shock_idx, shock_idx]))

        sim = self.simulate_obc(p0, eps, horizon=h+1)
        return sim

    def simulate_obc(self, p0, eps_path: np.ndarray, s0: Optional[np.ndarray] = None, horizon: Optional[int] = None):
        """OccBin-like deterministic path simulation with one constraint.

        Returns dict with keys: 'states' (ndarray), 'observables' (ndarray),
        'binding' (bool array), 'irf' (dict of DataFrames for states per shock if horizon small).
        """
        # Compile regime systems
        normal = self.compile_regime(self.normal_regime)
        binding = self.compile_regime(self.binding_regime)
        CCn, TTn, RRn, QQn, DDn, ZZn, HHn = normal.system_matrices(p0)
        CCb, TTB, RRB, QQb, DDb, ZZb, HHb = binding.system_matrices(p0)

        # Use normal regime's names to map variables to indices
        state_names: List[str] = normal.state_names
        obs_rows = len(ZZn)
        ns = TTn.shape[0]
        neps = RRn.shape[1]

        if horizon is None:
            horizon = eps_path.shape[0]
        H = int(horizon)
        eps = np.zeros((H, neps))
        eps[:min(H, eps_path.shape[0]), :min(neps, eps_path.shape[1])] = eps_path[:min(H, eps_path.shape[0]), :min(neps, eps_path.shape[1])]

        if s0 is None:
            s_prev = np.zeros((ns,), dtype=float)
        else:
            s_prev = np.asarray(s0, dtype=float).reshape((ns,))

        # Regime flags over time (binding True/False)
        bind = np.zeros((H,), dtype=bool)

        max_iter = 50
        prev_bind = None
        prev2_bind = None
        for _ in range(max_iter):
            # Simulate given current bind flags (block lower-triangular solve)
            states = np.zeros((H, ns), dtype=float)
            obs = np.zeros((H, obs_rows), dtype=float)
            s = s_prev.copy()
            for t in range(H):
                if bind[t]:
                    C, T, R, DD, ZZ = CCb, TTB, RRB, DDb, ZZb
                else:
                    C, T, R, DD, ZZ = CCn, TTn, RRn, DDn, ZZn
                s = C + T.dot(s) + R.dot(eps[t])
                states[t, :] = s
                obs[t, :] = (DD.T + ZZ.dot(s)).reshape((obs_rows,))

            # Evaluate when condition to update bind flags
            new_bind = self._evaluate_when_over_path(states, state_names)
            if np.array_equal(new_bind, bind):
                break
            # Anti-oscillation: if we detect a 2-cycle, prefer the union (more binding)
            if prev2_bind is not None and np.array_equal(new_bind, prev2_bind) and not np.array_equal(new_bind, bind):
                bind = np.logical_or(new_bind, bind)
                break
            prev2_bind = prev_bind
            prev_bind = bind.copy()
            bind = new_bind

        # Build a simple IRF dict (states only) keyed by shock names for convenience
        irf = {}
        for i, sh in enumerate(normal.shock_names):
            irf[sh] = None  # not building DataFrames here to avoid extra deps

        return {
            "states": states,
            "observables": obs,
            "binding": bind,
            "irf": irf,
            "state_names": state_names,
            "obs_names": normal.obs_names,
            "shock_names": normal.shock_names,
        }

    # --- Helpers ---
    def _compile_when(self, when: str) -> Tuple[List[str], Any]:
        if not when:
            return [], lambda *args: False
        expr = sp.sympify(when)
        syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
        lam = sp.lambdify([s for s in syms], expr, modules=["numpy"])
        names = [s.name for s in syms]
        return names, lam

    def _evaluate_when_over_path(self, states: np.ndarray, state_names: List[str]) -> np.ndarray:
        if not self._when_vars:
            return np.zeros((states.shape[0],), dtype=bool)
        idx = [state_names.index(n) for n in self._when_vars]
        vals = [states[:, j] for j in idx]
        out = self._when_lambda(*vals)
        return np.asarray(out, dtype=bool)


def read_obc(model_yaml: Dict[str, Any]) -> OccBinModel:
    """Construct an OccBinModel from YAML with 'regimes' and 'constraints'.

    Expects one constraint for now. Regime models are built by reusing DSGE.read
    with the same declarations/calibration but regime-specific equations.
    """
    dec = model_yaml["declarations"]
    cal = model_yaml["calibration"]

    regimes_yaml: Dict[str, Any] = model_yaml.get("regimes", {})
    constraints = model_yaml.get("constraints", [])
    if not regimes_yaml or not constraints:
        raise ValueError("OBC YAML must include 'regimes' and a non-empty 'constraints' list")

    # Build DSGE models per regime by reconstructing a minimal YAML per regime
    regimes: Dict[str, DSGE] = {}
    for name, reg in regimes_yaml.items():
        eqs = reg.get("equations") or reg
        if isinstance(eqs, dict):
            eq_block = {"equations": eqs}
        else:
            eq_block = {"equations": {"model": eqs, "observables": model_yaml.get("equations", {}).get("observables", {})}}

        reg_yaml = {
            "declarations": dec,
            "calibration": cal,
            **eq_block,
        }
        regimes[name] = DSGE.read(reg_yaml)

    # For now, support single constraint
    constraint = constraints[0]
    return OccBinModel(regimes=regimes, constraint=constraint, yaml=model_yaml)
