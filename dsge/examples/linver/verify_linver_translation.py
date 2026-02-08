from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from dsge import read_yaml
from dsge.read_mod import read_mod
from dsge.symbols import Parameter, Shock, Variable


def _names(symbols: Iterable[object]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        name = getattr(s, "name", None)
        out.append(str(name) if name is not None else str(s))
    return out


def _calibration_dict(model) -> dict[str, float]:
    cal = model.get("calibration", {})
    if isinstance(cal, dict) and "parameters" in cal and isinstance(cal["parameters"], dict):
        # Some models use nested calibration dicts.
        return {str(k): float(v) for k, v in cal["parameters"].items()}
    if isinstance(cal, dict):
        return {str(k): float(v) for k, v in cal.items()}
    return {}


def _equations_by_lhs(model) -> dict[str, object]:
    eqs = model.get("equations", [])
    out: dict[str, object] = {}
    for eq in eqs:
        lhs = getattr(eq, "lhs", None)
        if isinstance(lhs, Variable) and getattr(lhs, "date", 0) == 0:
            key = lhs.name
        else:
            key = str(lhs)
        if key in out:
            raise ValueError(f"Duplicate LHS key {key!r} in equations.")
        out[key] = eq
    return out


def _symbols_in_expr(expr) -> list[object]:
    # Include time-shifted variables and shocks as independent symbols.
    return sorted(expr.free_symbols, key=str)


def _make_subs(
    syms: list[object],
    *,
    rng: np.random.Generator,
    param_values: dict[str, float],
    scale: float,
) -> dict[object, float]:
    subs: dict[object, float] = {}
    for s in syms:
        if isinstance(s, Parameter):
            if s.name in param_values:
                subs[s] = float(param_values[s.name])
            else:
                # Uncalibrated parameter: draw something deterministic-ish.
                subs[s] = float(rng.normal())
        elif isinstance(s, (Variable, Shock)):
            subs[s] = float(scale * rng.normal())
        else:
            # Generic SymPy symbol
            subs[s] = float(scale * rng.normal())
    return subs


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify LINVER YAML translation vs a generated Dynare .mod file.")
    ap.add_argument("--yaml", dest="yaml_path", type=Path, required=True, help="Path to dsge YAML model.")
    ap.add_argument("--mod", dest="mod_path", type=Path, required=True, help="Path to generated Dynare .mod file.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for numeric equivalence checks.")
    ap.add_argument("--trials", type=int, default=5, help="Number of random trials per equation.")
    ap.add_argument("--tol", type=float, default=1e-8, help="Tolerance for numeric equation mismatch.")
    ap.add_argument("--scale", type=float, default=0.5, help="Stddev scale for random variable/shock draws.")
    ap.add_argument("--max-mismatches", type=int, default=20, help="Stop after reporting this many mismatches.")
    args = ap.parse_args()

    yaml_model = read_yaml(str(args.yaml_path))
    mod_model = read_mod(str(args.mod_path))

    yaml_vars = set(_names(yaml_model["var_ordering"]))
    mod_vars = set(_names(mod_model["var_ordering"]))
    yaml_shocks = set(_names(yaml_model["shk_ordering"]))
    mod_shocks = set(_names(mod_model["shk_ordering"]))
    yaml_params = set(_names(yaml_model["parameters"]))
    mod_params = set(_names(mod_model["parameters"]))

    print("=== Declarations diff ===")
    print(f"vars:   yaml={len(yaml_vars)} mod={len(mod_vars)}  missing_in_yaml={len(mod_vars - yaml_vars)} missing_in_mod={len(yaml_vars - mod_vars)}")
    print(f"shocks: yaml={len(yaml_shocks)} mod={len(mod_shocks)}  missing_in_yaml={len(mod_shocks - yaml_shocks)} missing_in_mod={len(yaml_shocks - mod_shocks)}")
    print(f"params: yaml={len(yaml_params)} mod={len(mod_params)}  missing_in_yaml={len(mod_params - yaml_params)} missing_in_mod={len(yaml_params - mod_params)}")

    if mod_vars - yaml_vars:
        print("Missing vars in YAML (first 30):", sorted(mod_vars - yaml_vars)[:30])
    if yaml_vars - mod_vars:
        print("Extra vars in YAML (first 30):", sorted(yaml_vars - mod_vars)[:30])

    print("\n=== Calibration diff (common params) ===")
    cal_yaml = _calibration_dict(yaml_model)
    cal_mod = _calibration_dict(mod_model)
    common_params = sorted(set(cal_yaml) & set(cal_mod))
    cal_mism = []
    for p in common_params:
        a = float(cal_yaml[p])
        b = float(cal_mod[p])
        if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 1e-12 * max(1.0, abs(a), abs(b)):
            cal_mism.append((p, a, b))
    print(f"common params: {len(common_params)}  mismatches: {len(cal_mism)}")
    for p, a, b in cal_mism[:20]:
        print(f"  {p}: yaml={a:.17g} mod={b:.17g}")

    # Use YAML calibration as the canonical parameter mapping for equivalence checks.
    # (If mod has extra params, they will be drawn randomly.)
    param_values = dict(cal_yaml)

    yaml_eq = _equations_by_lhs(yaml_model)
    mod_eq = _equations_by_lhs(mod_model)

    lhs_common = sorted(set(yaml_eq) & set(mod_eq))
    lhs_only_yaml = sorted(set(yaml_eq) - set(mod_eq))
    lhs_only_mod = sorted(set(mod_eq) - set(yaml_eq))

    print("\n=== Equation keys ===")
    print(f"eqs by LHS: yaml={len(yaml_eq)} mod={len(mod_eq)} common={len(lhs_common)}")
    if lhs_only_yaml:
        print("LHS only in YAML (first 30):", lhs_only_yaml[:30])
    if lhs_only_mod:
        print("LHS only in MOD (first 30):", lhs_only_mod[:30])

    rng = np.random.default_rng(int(args.seed))
    mismatches = 0

    print("\n=== Numeric equivalence checks ===")
    for lhs in lhs_common:
        e1 = yaml_eq[lhs].set_eq_zero
        e2 = mod_eq[lhs].set_eq_zero
        syms = _symbols_in_expr(e1) + [s for s in _symbols_in_expr(e2) if s not in _symbols_in_expr(e1)]
        # Dedup while preserving order
        seen = set()
        syms2 = []
        for s in syms:
            if s not in seen:
                syms2.append(s)
                seen.add(s)
        syms = syms2

        max_abs = 0.0
        for _ in range(int(args.trials)):
            subs = _make_subs(syms, rng=rng, param_values=param_values, scale=float(args.scale))
            v1 = float(e1.subs(subs))
            v2 = float(e2.subs(subs))
            max_abs = max(max_abs, abs(v1 - v2))
            if max_abs > float(args.tol):
                break

        if max_abs > float(args.tol):
            mismatches += 1
            print(f"Mismatch lhs={lhs!r} max|diff|={max_abs:.3g}")
            if mismatches >= int(args.max_mismatches):
                print("Stopping early (too many mismatches).")
                break

    if mismatches == 0:
        print("All common equations matched within tolerance.")
    else:
        print(f"Total mismatches: {mismatches} (of {len(lhs_common)} common equations).")


if __name__ == "__main__":
    main()

