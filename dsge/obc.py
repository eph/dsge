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

    def __init__(self, *, regimes: Dict[str, DSGE], constraint: Dict[str, Any] | List[Dict[str, Any]], yaml: Dict[str, Any]):
        self.regimes = regimes
        # Normalize constraints to a list
        if isinstance(constraint, dict):
            self.constraints: List[Dict[str, Any]] = [constraint]
        else:
            self.constraints = list(constraint)
        self.yaml = yaml

        # Borrow shared attributes from one regime (we require consistency)
        any_regime = next(iter(regimes.values()))
        self.parameters = any_regime.parameters
        self.shocks = any_regime.shocks
        self.variables = any_regime.variables
        self.observables = any_regime["observables"]

        # Prepare convenience names (fallbacks if YAML doesn't provide 'normal')
        if 'normal' in regimes:
            self.normal_regime = 'normal'
        else:
            nr = [c.get('normal_regime') for c in self.constraints if c.get('normal_regime')]
            self.normal_regime = nr[0] if nr else next(iter(regimes.keys()))
        br = [c.get('binding_regime') for c in self.constraints if c.get('binding_regime')]
        self.binding_regime = br[0] if br else self.normal_regime
        self.horizon_default = max(int(c.get("horizon", 40)) for c in self.constraints)

        # Parse each 'when' condition to a fast lambda
        self._when_exprs = [c.get("when", "") for c in self.constraints]
        self._when_compiled = [self._compile_when(expr) for expr in self._when_exprs]

    def p0(self):
        return list(map(lambda x: self.yaml["calibration"]["parameters"][str(x)], self.parameters))

    def compile_regime(self, name: str):
        return self.regimes[name].compile_model()

    def compile_model(self):
        """Compile both regimes and return a compiled OBC model.

        Aligns with the mod->compile_model pattern used elsewhere.
        """
        return OccBinCompiled(self)

    # Back-compat convenience; prefer compiled.impulse_response
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

    # New aligned naming
    def impulse_response(self, p0, h: int = 20, shock_name: Optional[str] = None):
        return self.compile_model().impulse_response(p0, h=h, shock_name=shock_name)

    def simulate(self, p0, eps_path: np.ndarray, horizon: Optional[int] = None, s0: Optional[np.ndarray] = None):
        return self.compile_model().simulate(p0, eps_path=eps_path, horizon=horizon, s0=s0)

    # Standardized wrappers
    def impulse_response_states(self, p0, h: int = 20, shock_name: Optional[str] = None):
        return self.compile_model().impulse_response_states(p0, h=h, shock_name=shock_name)

    def impulse_response_observables(self, p0, h: int = 20, shock_name: Optional[str] = None):
        return self.compile_model().impulse_response_observables(p0, h=h, shock_name=shock_name)

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

        # Regime flags over time (binding True/False) and chosen regimes
        bind = np.zeros((H,), dtype=bool)
        regimes_used: List[str] = [self.normal_regime] * H

        max_iter = 50
        prev_bind = None
        prev2_bind = None
        for _ in range(max_iter):
            # Simulate given current bind flags (block lower-triangular solve)
            states = np.zeros((H, ns), dtype=float)
            obs = np.zeros((H, obs_rows), dtype=float)
            s = s_prev.copy()
            for t in range(H):
                reg = regimes_used[t]
                if reg == self.binding_regime:
                    C, T, R, DD, ZZ = CCb, TTB, RRB, DDb, ZZb
                elif reg == self.normal_regime:
                    C, T, R, DD, ZZ = CCn, TTn, RRn, DDn, ZZn
                else:
                    model_t = self.compile_regime(reg)
                    C, T, R, QQt, DD, ZZ, HHt = model_t.system_matrices(p0)
                s = C + T.dot(s) + R.dot(eps[t])
                states[t, :] = s
                obs[t, :] = (DD.T + ZZ.dot(s)).reshape((obs_rows,))

            # Evaluate when condition to update bind flags
            active_mat = self._evaluate_multi_when(states, state_names)
            new_bind = np.any(active_mat, axis=1) if active_mat.size else np.zeros((H,), dtype=bool)
            # Update chosen regimes per time based on active constraints
            regimes_new: List[str] = []
            for t in range(H):
                active = [self.constraints[i]['name'] for i, b in enumerate(active_mat[t, :]) if b]
                if len(active) == 0:
                    regimes_new.append(self.normal_regime)
                elif len(active) == 1:
                    c = next(c for c in self.constraints if c['name'] == active[0])
                    regimes_new.append(c.get('binding_regime', self.binding_regime))
                else:
                    key = ','.join(sorted(active))
                    if key in self.regime_map:
                        regimes_new.append(self.regime_map[key])
                    else:
                        raise NotImplementedError(f"Multiple active constraints {active} but no regime_map entry for key '{key}'")
            if np.array_equal(new_bind, bind):
                regimes_used = regimes_new
                break
            # Anti-oscillation: if we detect a 2-cycle, prefer the union (more binding)
            if prev2_bind is not None and np.array_equal(new_bind, prev2_bind) and not np.array_equal(new_bind, bind):
                bind = np.logical_or(new_bind, bind)
                # Prefer binding regimes where union true
                for t in range(H):
                    if bind[t]:
                        # keep or set to binding default if previously normal
                        if regimes_new[t] == self._obc.normal_regime:
                            regimes_new[t] = self._obc.binding_regime
                regimes_used = regimes_new
                break
            # Hysteresis: require two consecutive False before releasing a binding position
            if prev_bind is not None:
                release_candidates = np.logical_and(bind, np.logical_not(new_bind))
                sticky = np.logical_and(release_candidates, prev_bind)
                new_bind = np.where(sticky, True, new_bind)
                # Keep previous regime on sticky positions
                for t in range(H):
                    if sticky[t]:
                        regimes_new[t] = regimes_used[t]
            prev2_bind = prev_bind
            prev_bind = bind.copy()
            bind = new_bind
            regimes_used = regimes_new

        # Build a simple IRF dict (states only) keyed by shock names for convenience
        irf = {}
        for i, sh in enumerate(normal.shock_names):
            irf[sh] = None  # not building DataFrames here to avoid extra deps

        return {
            "states": states,
            "observables": obs,
            "binding": bind,
            "regimes": regimes_used,
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

    def _evaluate_multi_when(self, states: np.ndarray, state_names: List[str]) -> np.ndarray:
        H = states.shape[0]
        if not self._when_compiled:
            return np.zeros((H, 0), dtype=bool)
        out = np.zeros((H, len(self._when_compiled)), dtype=bool)
        for k, (names, lam) in enumerate(self._when_compiled):
            if not names:
                continue
            idx = [state_names.index(n) for n in names]
            vals = [states[:, j] for j in idx]
            cond = lam(*vals)
            out[:, k] = np.asarray(cond, dtype=bool)
        return out


class OccBinCompiled:
    """Compiled OBC model aligning with the usual compiled interface.

    Provides impulse_response and simulate methods. Holds references to the
    parent OccBinModel for constraint evaluation.
    """

    def __init__(self, obc: OccBinModel):
        self._obc = obc
        # Compile regime models once; system matrices still depend on p0
        self._normal = obc.compile_regime(obc.normal_regime)
        self._binding = obc.compile_regime(obc.binding_regime)
        self.shock_names = self._normal.shock_names
        self.state_names = self._normal.state_names
        self.obs_names = self._normal.obs_names

    def impulse_response(self, p0, h: int = 20, shock_name: Optional[str] = None):
        CC, TTn, RRn, QQn, DDn, ZZn, HHn = self._normal.system_matrices(p0)
        neps = RRn.shape[1]
        if shock_name is None:
            shock_idx = 0
        else:
            shock_idx = self.shock_names.index(shock_name)
        eps = np.zeros((h+1, neps))
        eps[0, shock_idx] = float(np.sqrt(QQn[shock_idx, shock_idx]))
        res = self.simulate(p0, eps_path=eps, horizon=h+1)
        # Build DataFrames for states/observables IRFs
        try:
            import pandas as pd
            res['states_df'] = pd.DataFrame(res['states'], columns=self.state_names)
            res['observables_df'] = pd.DataFrame(res['observables'], columns=self.obs_names)
        except Exception:
            pass
        return res

    # Standardized interfaces matching LinearDSGEModel style
    def impulse_response_states(self, p0, h: int = 20, shock_name: Optional[str] = None):
        """Return dict shock->DataFrame of state IRFs over horizon h+1."""
        CC, TTn, RRn, QQn, DDn, ZZn, HHn = self._normal.system_matrices(p0)
        shock_list = [shock_name] if shock_name else self.shock_names
        out = {}
        try:
            import pandas as pd
        except Exception:
            pd = None  # type: ignore
        for sh in shock_list:
            i = self.shock_names.index(sh)
            eps = np.zeros((h+1, RRn.shape[1]))
            eps[0, i] = float(np.sqrt(QQn[i, i]))
            res = self.simulate(p0, eps_path=eps, horizon=h+1)
            if pd is not None:
                out[sh] = pd.DataFrame(res['states'], columns=self.state_names)
            else:
                out[sh] = res['states']
        return out

    def impulse_response_observables(self, p0, h: int = 20, shock_name: Optional[str] = None):
        """Return dict shock->DataFrame of observable IRFs over horizon h+1."""
        CC, TTn, RRn, QQn, DDn, ZZn, HHn = self._normal.system_matrices(p0)
        shock_list = [shock_name] if shock_name else self.shock_names
        out = {}
        try:
            import pandas as pd
        except Exception:
            pd = None  # type: ignore
        for sh in shock_list:
            i = self.shock_names.index(sh)
            eps = np.zeros((h+1, RRn.shape[1]))
            eps[0, i] = float(np.sqrt(QQn[i, i]))
            res = self.simulate(p0, eps_path=eps, horizon=h+1)
            if pd is not None:
                out[sh] = pd.DataFrame(res['observables'], columns=self.obs_names)
            else:
                out[sh] = res['observables']
        return out

    def simulate(self, p0, eps_path: np.ndarray, horizon: Optional[int] = None, s0: Optional[np.ndarray] = None):
        # Evaluate per-regime matrices at p0
        CCn, TTn, RRn, QQn, DDn, ZZn, HHn = self._normal.system_matrices(p0)
        CCb, TTB, RRB, QQb, DDb, ZZb, HHb = self._binding.system_matrices(p0)

        ns = TTn.shape[0]
        neps = RRn.shape[1]
        nobs = ZZn.shape[0]

        if horizon is None:
            horizon = eps_path.shape[0]
        H = int(horizon)
        eps = np.zeros((H, neps))
        eps[:min(H, eps_path.shape[0]), :min(neps, eps_path.shape[1])] = eps_path[:min(H, eps_path.shape[0]), :min(neps, eps_path.shape[1])]

        s_prev = np.zeros((ns,), dtype=float) if s0 is None else np.asarray(s0, dtype=float).reshape((ns,))
        bind = np.zeros((H,), dtype=bool)
        regimes_used: List[str] = [self._obc.normal_regime] * H

        max_iter = 50
        prev_bind = None
        prev2_bind = None
        for _ in range(max_iter):
            states = np.zeros((H, ns), dtype=float)
            obs = np.zeros((H, nobs), dtype=float)
            s = s_prev.copy()
            for t in range(H):
                reg = regimes_used[t]
                if reg == self._obc.binding_regime:
                    C, T, R, DD, ZZ = CCb, TTB, RRB, DDb, ZZb
                elif reg == self._obc.normal_regime:
                    C, T, R, DD, ZZ = CCn, TTn, RRn, DDn, ZZn
                else:
                    # Composite or specific regime
                    model_t = self._obc.compile_regime(reg)
                    C, T, R, QQt, DD, ZZ, HHt = model_t.system_matrices(p0)
                s = C + T.dot(s) + R.dot(eps[t])
                states[t, :] = s
                obs[t, :] = (DD.T + ZZ.dot(s)).reshape((nobs,))

            active_mat = self._obc._evaluate_multi_when(states, self.state_names)
            new_bind = np.any(active_mat, axis=1) if active_mat.size else np.zeros((H,), dtype=bool)
            # Build proposed regimes for this iteration
            regimes_new: List[str] = []
            for t in range(H):
                active = [self._obc.constraints[i]['name'] for i, b in enumerate(active_mat[t, :]) if b]
                if len(active) == 0:
                    regimes_new.append(self._obc.normal_regime)
                elif len(active) == 1:
                    c = next(c for c in self._obc.constraints if c['name'] == active[0])
                    regimes_new.append(c.get('binding_regime', self._obc.binding_regime))
                else:
                    key = ','.join(sorted(active))
                    if key in self._obc.regime_map:
                        regimes_new.append(self._obc.regime_map[key])
                    else:
                        raise NotImplementedError(f"Multiple active constraints {active} but no regime_map entry for key '{key}'")
            # Convergence checks
            if np.array_equal(new_bind, bind):
                regimes_used = regimes_new
                break
            if prev2_bind is not None and np.array_equal(new_bind, prev2_bind) and not np.array_equal(new_bind, bind):
                bind = np.logical_or(new_bind, bind)
                regimes_used = regimes_new
                break
            prev2_bind = prev_bind
            prev_bind = bind.copy()
            bind = new_bind
            regimes_used = regimes_new

        return {
            "states": states,
            "observables": obs,
            "binding": bind,
            "regimes": regimes_used,
            "state_names": self.state_names,
            "obs_names": self.obs_names,
            "shock_names": self.shock_names,
        }


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

    # Pass through full constraints list
    return OccBinModel(regimes=regimes, constraint=constraints, yaml=model_yaml)
