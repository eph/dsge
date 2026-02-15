from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp

from .endogenous_horizon_switching import EndogenousHorizonSwitchingModel


def _expr_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return str(float(x))
    return str(x)


def _symbol_dict(names: Sequence[str]) -> Dict[str, sp.Symbol]:
    return {str(n): sp.Symbol(str(n)) for n in names}


def _base_parse_context() -> Dict[str, Any]:
    return {
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "Max": sp.Max,
        "Min": sp.Min,
        "oo": sp.oo,
    }


def _parse_expr(
    expr: Any,
    *,
    ctx: Mapping[str, Any],
    allowed_symbols: Iterable[sp.Symbol],
    where: str,
) -> sp.Expr:
    s = _expr_str(expr)
    try:
        out = sp.sympify(s, locals=dict(ctx))
    except Exception as e:  # pragma: no cover - error formatting
        raise ValueError(f"While parsing {where} expression {s!r}: {e}") from e

    allowed = set(allowed_symbols)
    unknown = [sym for sym in out.free_symbols if sym not in allowed]
    if unknown:
        raise ValueError(f"Unknown symbol(s) in {where} expression {s!r}: {[str(u) for u in unknown]}")
    return out


def _toposort_definitions(def_exprs: Mapping[sp.Symbol, sp.Expr]) -> list[sp.Symbol]:
    # Kahn topological sort.
    deps: Dict[sp.Symbol, set[sp.Symbol]] = {}
    sym_set = set(def_exprs.keys())
    for k, v in def_exprs.items():
        deps[k] = set(sym for sym in v.free_symbols if sym in sym_set)

    ready = sorted([k for k, d in deps.items() if not d], key=lambda s: s.name)
    order: list[sp.Symbol] = []

    while ready:
        n = ready.pop(0)
        order.append(n)
        for k in list(deps.keys()):
            if n in deps[k]:
                deps[k].remove(n)
                if not deps[k]:
                    ready.append(k)
                    ready.sort(key=lambda s: s.name)
        deps.pop(n, None)

    if deps:
        cyc = sorted([k.name for k in deps.keys()])
        raise ValueError(f"Cycle detected in model.definitions: {cyc}")
    return order


def _parse_definitions(
    raw_defs: Mapping[str, Any],
    *,
    ctx: Dict[str, Any],
    allowed_symbols: set[sp.Symbol],
) -> Dict[sp.Symbol, sp.Expr]:
    if not raw_defs:
        return {}

    def_syms = {name: sp.Symbol(str(name)) for name in raw_defs.keys()}
    ctx.update(def_syms)
    allowed_symbols = set(allowed_symbols) | set(def_syms.values())

    parsed: Dict[sp.Symbol, sp.Expr] = {}
    for name, expr in raw_defs.items():
        sym = def_syms[str(name)]
        parsed[sym] = _parse_expr(expr, ctx=ctx, allowed_symbols=allowed_symbols, where=f"model.definitions.{name}")

    order = _toposort_definitions(parsed)
    resolved: Dict[sp.Symbol, sp.Expr] = {}
    for sym in order:
        resolved[sym] = sp.simplify(parsed[sym].subs(resolved))

    return resolved


def _parse_matrix(
    raw: Sequence[Sequence[Any]],
    *,
    ctx: Mapping[str, Any],
    allowed_symbols: Iterable[sp.Symbol],
    where: str,
) -> sp.Matrix:
    rows = [list(r) for r in raw]
    if not rows:
        raise ValueError(f"{where} must be a non-empty 2D list.")
    ncol = len(rows[0])
    if ncol == 0:
        raise ValueError(f"{where} must have at least one column.")
    for i, r in enumerate(rows):
        if len(r) != ncol:
            raise ValueError(f"{where} must be rectangular; row 0 has {ncol} cols but row {i} has {len(r)}.")

    mat = sp.Matrix(
        [
            [
                _parse_expr(x, ctx=ctx, allowed_symbols=allowed_symbols, where=f"{where}[{i},{j}]")
                for j, x in enumerate(r)
            ]
            for i, r in enumerate(rows)
        ]
    )
    return mat


def _parse_vector(
    raw: Sequence[Any],
    *,
    ctx: Mapping[str, Any],
    allowed_symbols: Iterable[sp.Symbol],
    where: str,
) -> sp.Matrix:
    if raw is None:
        raise ValueError(f"{where} is required.")
    vals = list(raw)
    if not vals:
        raise ValueError(f"{where} must be non-empty.")
    return sp.Matrix(
        [
            _parse_expr(x, ctx=ctx, allowed_symbols=allowed_symbols, where=f"{where}[{i}]")
            for i, x in enumerate(vals)
        ]
    )


def _require_shape(arr: np.ndarray, shape: Tuple[int, ...], *, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return arr


@dataclass(frozen=True)
class SwitchingSSMSpec:
    components: list[str]
    state_names: list[str]
    shock_names: list[str]
    obs_names: list[str]
    parameter_names: list[str]
    p0: np.ndarray


def read_switching_ssm(model_yaml: Mapping[str, Any]) -> EndogenousHorizonSwitchingModel:
    dec = model_yaml["declarations"]
    model = model_yaml["model"]

    components = [str(c) for c in dec["components"]]
    state_names = [str(s) for s in dec["states"]]
    shock_names = [str(s) for s in dec["shocks"]]
    obs_names = [str(o) for o in dec["observables"]]
    parameter_names = [str(p) for p in dec.get("parameters", [])]

    if len(set(components)) != len(components):
        raise ValueError(f"Duplicate components: {components}")
    if len(set(state_names)) != len(state_names):
        raise ValueError(f"Duplicate states: {state_names}")
    if len(set(obs_names)) != len(obs_names):
        raise ValueError(f"Duplicate observables: {obs_names}")

    nstate = len(state_names)
    nshock = len(shock_names)
    nobs = len(obs_names)

    # Calibration / parameter ordering
    cal = model_yaml.get("calibration", {}) or {}
    cal_params = (cal.get("parameters", {}) or {}) if isinstance(cal, dict) else {}
    missing = [p for p in parameter_names if p not in cal_params]
    if missing:
        raise ValueError(f"Missing calibration.parameters entries for: {missing}")
    p0 = np.array([float(cal_params[p]) for p in parameter_names], dtype=float)

    # Symbolic context
    param_syms = _symbol_dict(parameter_names)
    state_syms = _symbol_dict(state_names)
    obs_syms = _symbol_dict(obs_names)
    regime_syms = {f"k_{c}": sp.Symbol(f"k_{c}", integer=True) for c in components}

    ctx: Dict[str, Any] = {}
    ctx.update(_base_parse_context())
    ctx.update(param_syms)
    ctx.update(state_syms)
    ctx.update(obs_syms)
    ctx.update(regime_syms)

    allowed_symbols = set(param_syms.values()) | set(state_syms.values()) | set(obs_syms.values()) | set(regime_syms.values())

    # Definitions
    def_subs = _parse_definitions(model.get("definitions", {}) or {}, ctx=ctx, allowed_symbols=allowed_symbols)
    allowed_symbols = allowed_symbols | set(def_subs.keys())

    # Matrices (symbolic)
    TT_expr = _parse_matrix(model["TT"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.TT").subs(def_subs)
    RR_expr = _parse_matrix(model["RR"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.RR").subs(def_subs)
    ZZ_expr = _parse_matrix(model["ZZ"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.ZZ").subs(def_subs)
    DD_expr = _parse_vector(model["DD"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.DD").subs(def_subs)
    QQ_expr = _parse_matrix(model["QQ"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.QQ").subs(def_subs)
    HH_expr = _parse_matrix(model["HH"], ctx=ctx, allowed_symbols=allowed_symbols, where="model.HH").subs(def_subs)

    # Validate that system matrices don't depend on state/obs symbols.
    forbidden = set(state_syms.values()) | set(obs_syms.values())
    for name, expr in [
        ("TT", TT_expr),
        ("RR", RR_expr),
        ("ZZ", ZZ_expr),
        ("DD", DD_expr),
        ("QQ", QQ_expr),
        ("HH", HH_expr),
    ]:
        bad = sorted([s.name for s in expr.free_symbols if s in forbidden])
        if bad:
            raise ValueError(f"model.{name} expressions must not depend on state/observable symbols, found: {bad}")

    # Lambdify system matrices as functions of (params, regime).
    arg_syms = [param_syms[p] for p in parameter_names] + [regime_syms[f"k_{c}"] for c in components]
    TT_func = sp.lambdify(arg_syms, TT_expr, modules="numpy")
    RR_func = sp.lambdify(arg_syms, RR_expr, modules="numpy")
    ZZ_func = sp.lambdify(arg_syms, ZZ_expr, modules="numpy")
    DD_func = sp.lambdify(arg_syms, DD_expr, modules="numpy")
    QQ_func = sp.lambdify(arg_syms, QQ_expr, modules="numpy")
    HH_func = sp.lambdify(arg_syms, HH_expr, modules="numpy")

    def solve_given_regime(params_vec: np.ndarray, regime: Tuple[int, ...]):
        params_vec = np.asarray(params_vec, dtype=float).reshape(-1)
        if params_vec.size != len(parameter_names):
            raise ValueError(
                f"params must have length {len(parameter_names)} (parameters={parameter_names}), got {params_vec.size}"
            )
        regime = tuple(int(x) for x in regime)
        if len(regime) != len(components):
            raise ValueError(f"regime must have length {len(components)}, got {len(regime)}")

        args = [*params_vec.tolist(), *[int(x) for x in regime]]
        TT = _require_shape(np.asarray(TT_func(*args), dtype=float), (nstate, nstate), name="TT")
        RR = _require_shape(np.asarray(RR_func(*args), dtype=float), (nstate, nshock), name="RR")
        ZZ = _require_shape(np.asarray(ZZ_func(*args), dtype=float), (nobs, nstate), name="ZZ")
        DD = _require_shape(np.asarray(DD_func(*args), dtype=float).reshape(-1), (nobs,), name="DD")
        QQ = _require_shape(np.asarray(QQ_func(*args), dtype=float), (nshock, nshock), name="QQ")
        HH = _require_shape(np.asarray(HH_func(*args), dtype=float), (nobs, nobs), name="HH")
        return TT, RR, ZZ, DD, QQ, HH

    # Horizon choice config
    hc = dec["horizon_choice"]
    hc_components = hc.get("components", {}) or {}
    if set(hc_components.keys()) != set(components):
        raise ValueError(
            "declarations.horizon_choice.components keys must match declarations.components. "
            f"Got {sorted(hc_components.keys())}, expected {sorted(components)}."
        )

    selection_order = hc.get("selection_order", None)

    k_max: Dict[str, int] = {}
    cost_params: Dict[str, Tuple[float, float]] = {}
    lam: Dict[str, float] = {}
    policy_exprs: Dict[str, sp.Expr] = {}

    allowed_policy_symbols = set(param_syms.values()) | set(state_syms.values()) | set(obs_syms.values())
    policy_arg_syms = [param_syms[p] for p in parameter_names] + [state_syms[s] for s in state_names] + [obs_syms[o] for o in obs_names]

    lam_ctx: Dict[str, Any] = dict(_base_parse_context())
    lam_ctx.update(param_syms)
    lam_allowed = set(param_syms.values())
    lam_arg_syms = [param_syms[p] for p in parameter_names]

    lam_func_by_comp: Dict[str, Any] = {}
    a_func_by_comp: Dict[str, Any] = {}

    for comp in components:
        cfg = hc_components[comp]
        k_max[comp] = int(cfg["k_max"])
        if k_max[comp] < 0:
            raise ValueError(f"horizon_choice.components.{comp}.k_max must be >= 0, got {k_max[comp]}")

        # Cost: for now, constant marginal costs Δτ_{k+1} = a.
        a_expr = _parse_expr(cfg["cost"]["a"], ctx=lam_ctx, allowed_symbols=lam_allowed, where=f"horizon_choice.components.{comp}.cost.a")
        a_func_by_comp[comp] = sp.lambdify(lam_arg_syms, a_expr, modules="numpy")
        a_val = float(a_func_by_comp[comp](*p0.tolist()))
        cost_params[comp] = (a_val, 0.0)

        lam_expr = _parse_expr(cfg["lambda"], ctx=lam_ctx, allowed_symbols=lam_allowed, where=f"horizon_choice.components.{comp}.lambda")
        lam_func_by_comp[comp] = sp.lambdify(lam_arg_syms, lam_expr, modules="numpy")
        lam_val = float(lam_func_by_comp[comp](*p0.tolist()))
        lam[comp] = lam_val

        policy_exprs[comp] = _parse_expr(
            cfg["policy_object"],
            ctx=ctx,
            allowed_symbols=allowed_policy_symbols,
            where=f"horizon_choice.components.{comp}.policy_object",
        )

    policy_funcs = {c: sp.lambdify(policy_arg_syms, policy_exprs[c], modules="numpy") for c in components}

    # Info function exposes state values by name.
    def info_func(x_t: np.ndarray, t: int, chosen):
        x_t = np.asarray(x_t, dtype=float).reshape(-1)
        if x_t.shape != (nstate,):
            raise ValueError(f"x_t must have shape ({nstate},), got {x_t.shape}")
        info = {"x": x_t, "t": int(t), "chosen": dict(chosen)}
        for i, nm in enumerate(state_names):
            info[nm] = float(x_t[i])
        return info

    model_ref: Dict[str, Any] = {}

    def policy_object(params_vec: np.ndarray, info_t: Mapping[str, Any], component: str, k: int, chosen_regime):
        m = model_ref.get("model")
        if m is None:  # pragma: no cover - defensive
            raise RuntimeError("Internal error: model not initialized in policy_object closure.")

        params_vec = np.asarray(params_vec, dtype=float).reshape(-1)
        x_t = np.asarray(info_t["x"], dtype=float).reshape(-1)

        k_by_comp = {c: int(chosen_regime.get(c, 0)) for c in components}
        k_by_comp[component] = int(k)
        regime = tuple(k_by_comp[c] for c in components)

        TT, RR, ZZ, DD, QQ, HH = m.get_mats(params_vec, regime)
        y_hat = ZZ @ x_t + np.asarray(DD, dtype=float).reshape(-1)

        args = [*params_vec.tolist(), *x_t.tolist(), *y_hat.tolist()]
        out = policy_funcs[component](*args)
        return np.asarray(out, dtype=float)

    out_model = EndogenousHorizonSwitchingModel(
        components=components,
        k_max=k_max,
        cost_params=cost_params,
        lam=lam,
        cost_func=lambda params_vec, component: (float(a_func_by_comp[str(component)](*np.asarray(params_vec, dtype=float).reshape(-1).tolist())), 0.0),
        lam_func=lambda params_vec, component: float(lam_func_by_comp[str(component)](*np.asarray(params_vec, dtype=float).reshape(-1).tolist())),
        solve_given_regime=solve_given_regime,
        policy_object=policy_object,
        info_func=info_func,
        selection_order=selection_order,
    )

    model_ref["model"] = out_model

    # Attach metadata (best-effort, for downstream tooling/tests).
    setattr(out_model, "spec", SwitchingSSMSpec(components, state_names, shock_names, obs_names, parameter_names, p0))
    setattr(out_model, "parameter_names", list(parameter_names))
    setattr(out_model, "state_names", list(state_names))
    setattr(out_model, "shock_names", list(shock_names))
    setattr(out_model, "obs_names", list(obs_names))
    setattr(out_model, "p0", p0)

    return out_model
