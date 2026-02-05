import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

from dsge.dynare_export import to_dynare_mod
from dsge.parse_yaml import read_yaml


def _dynare_cmd() -> str | None:
    return shutil.which("dynare") or shutil.which("dynare-octave")


def _find_dynare_root(dynare_cmd: str) -> str | None:
    env_root = os.environ.get("DYNARE_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "matlab" / "dynare.m").exists():
            return str(p)

    cmd_path = Path(dynare_cmd)
    try:
        default_root = cmd_path.resolve().parent.parent
        if (default_root / "matlab" / "dynare.m").exists():
            return str(default_root)
    except Exception:
        pass

    home = Path.home()
    downloads = home / "Downloads"
    if downloads.is_dir():
        for child in sorted(downloads.glob("dynare-*")):
            if (child / "matlab" / "dynare.m").exists():
                return str(child)

    for p in (home / "dynare", home / "Dynare"):
        if (p / "matlab" / "dynare.m").exists():
            return str(p)

    return None


def _matlab_char_array_to_strings(x) -> list[str]:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S"} and arr.ndim == 1:
        return [str(s).strip() for s in arr.tolist()]
    if arr.dtype.kind in {"U", "S"} and arr.ndim == 2:
        return ["".join(row).strip() for row in arr.tolist()]
    if arr.dtype == object:
        out: list[str] = []
        for item in arr.ravel().tolist():
            out.extend(_matlab_char_array_to_strings(item))
        return out
    return [str(arr).strip()]


def _get_struct_field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise AttributeError(name)


def _ensure_2d(x, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape == (rows, cols):
        return arr
    if arr.ndim == 0:
        return arr.reshape(rows, cols)
    if arr.ndim == 1:
        return arr.reshape(rows, cols)
    return arr.reshape(rows, cols)


@pytest.mark.skipif(_dynare_cmd() is None, reason="Dynare not installed (expected `dynare` on PATH).")
@pytest.mark.skipif(os.environ.get("DSGE_RUN_DYNARE") != "1", reason="Set DSGE_RUN_DYNARE=1 to run Dynare checks.")
def test_second_order_matches_dynare_for_small_nonlinear_model():
    """
    Cross-check our 2nd-order solver against Dynare on a tiny benchmark.

    The model is in deviation form with steady state at 0:

      x = rho*x(-1) + e
      y = beta*y(+1) + x + 0.5*phi*x^2
    """
    yaml_text = """
declarations:
  name: dynare_so_check
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta, phi]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x + phi/2*x^2
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
    phi: 0.5
  covariance:
    e: 0.01
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    ours = m.solve_second_order(p0).as_dynare_like()

    dynare_mod = to_dynare_mod(m, order=2, pruning=True, irf=0)

    dynare = _dynare_cmd()
    assert dynare is not None

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod_path = td_path / "dynare_so_check.mod"
        mod_path.write_text(dynare_mod.mod_text, encoding="utf-8")

        env = os.environ.copy()
        dynare_root = _find_dynare_root(dynare)
        if dynare_root is not None:
            env.setdefault("DYNARE_ROOT", dynare_root)

        proc = subprocess.run(
            [dynare, str(mod_path), "noclearall", "nolog", "nograph"],
            cwd=td_path,
            capture_output=True,
            text=True,
            timeout=240,
            env=env,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "Dynare run failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        results_candidates = sorted(td_path.rglob("dynare_so_check_results.mat"))
        assert results_candidates, f"Expected results file not found under: {td_path}"
        results_path = results_candidates[0]

        mat = loadmat(results_path, squeeze_me=True, struct_as_record=False)
        oo_ = mat["oo_"]
        M_ = mat["M_"]

        endo_names_decl = _matlab_char_array_to_strings(_get_struct_field(M_, "endo_names"))
        exo_names = _matlab_char_array_to_strings(_get_struct_field(M_, "exo_names"))

        # Dynare decision-rule arrays are in `dr.order_var` ordering (not necessarily declaration ordering).
        order_var = np.asarray(_get_struct_field(oo_.dr, "order_var")).astype(int).ravel()
        if order_var.size:
            dyn_row_names = [endo_names_decl[i - 1] for i in order_var]  # 1-based -> 0-based
        else:
            dyn_row_names = list(endo_names_decl)

        state_var = np.asarray(_get_struct_field(oo_.dr, "state_var")).astype(int).ravel()
        # `state_var` indexes endogenous variables in declaration ordering.
        state_names = [endo_names_decl[i - 1] for i in state_var]

        nendo = len(dyn_row_names)
        nstate = len(state_var)
        nexo = len(exo_names)

        # Reorder ours to Dynare's (endo, state) ordering.
        ours_endo_idx = [ours["endo_names"].index(n) for n in dyn_row_names]

        def _state_base(n: str) -> str:
            return n[:-5] if n.endswith("_LAG1") else n

        ours_state_idx = []
        for dn in state_names:
            candidates = [i for i, sn in enumerate(ours["state_names"]) if _state_base(sn) == dn or sn == dn]
            if not candidates:
                raise AssertionError(f"Could not map Dynare state '{dn}' into our state_names={ours['state_names']}")
            ours_state_idx.append(candidates[0])

        ours_ghx = ours["ghx"][ours_endo_idx, :][:, ours_state_idx]
        ours_ghu = ours["ghu"][ours_endo_idx, :]
        ours_ghxx = ours["ghxx"][ours_endo_idx, :]
        ours_ghxu = ours["ghxu"][ours_endo_idx, :]
        ours_ghuu = ours["ghuu"][ours_endo_idx, :]
        ours_ghs2 = ours["ghs2"][ours_endo_idx, :]

        dyn_ghx = _ensure_2d(_get_struct_field(oo_.dr, "ghx"), nendo, nstate)
        dyn_ghu = _ensure_2d(_get_struct_field(oo_.dr, "ghu"), nendo, nexo)
        dyn_ghxx = _ensure_2d(_get_struct_field(oo_.dr, "ghxx"), nendo, nstate * nstate)
        dyn_ghxu = _ensure_2d(_get_struct_field(oo_.dr, "ghxu"), nendo, nstate * nexo)
        dyn_ghuu = _ensure_2d(_get_struct_field(oo_.dr, "ghuu"), nendo, nexo * nexo)
        dyn_ghs2 = _ensure_2d(_get_struct_field(oo_.dr, "ghs2"), nendo, 1)

        assert_allclose(ours_ghx, dyn_ghx, rtol=5e-6, atol=5e-8)
        assert_allclose(ours_ghu, dyn_ghu, rtol=5e-6, atol=5e-8)
        assert_allclose(ours_ghxx, dyn_ghxx, rtol=5e-5, atol=5e-7)
        assert_allclose(ours_ghxu, dyn_ghxu, rtol=5e-5, atol=5e-7)
        assert_allclose(ours_ghuu, dyn_ghuu, rtol=5e-5, atol=5e-7)

        # Empirically (Dynare 6.x), `ghs2` matches the constant term coefficient at sigma=1.
        assert_allclose(dyn_ghs2, ours_ghs2, rtol=5e-5, atol=5e-7)


@pytest.mark.skipif(_dynare_cmd() is None, reason="Dynare not installed (expected `dynare` on PATH).")
@pytest.mark.skipif(os.environ.get("DSGE_RUN_DYNARE") != "1", reason="Set DSGE_RUN_DYNARE=1 to run Dynare checks.")
def test_second_order_matches_dynare_for_two_state_two_shock_model():
    """
    Cross-check order-2 arrays on a slightly richer toy model.

    Deviation form with steady state at 0:

      x = rho*x(-1) + e1
      z = rho_z*z(-1) + e2
      y = beta*y(+1) + x + z + 0.5*phi*x*z
    """
    yaml_text = """
declarations:
  name: dynare_so_check_2
  variables: [x, y, z]
  shocks: [e1, e2]
  parameters: [rho, rho_z, beta, phi]

equations:
  model:
    - x = rho*x(-1) + e1
    - z = rho_z*z(-1) + e2
    - y = beta*y(1) + x + z + phi/2*x*z
  observables:
    x: x
    y: y
    z: z

calibration:
  parameters:
    rho: 0.8
    rho_z: 0.7
    beta: 0.99
    phi: 0.4
  covariance:
    e1: 0.01
    e2: 0.02
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    ours = m.solve_second_order(p0).as_dynare_like()

    dynare_mod = to_dynare_mod(m, order=2, pruning=True, irf=0)

    dynare = _dynare_cmd()
    assert dynare is not None

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod_path = td_path / "dynare_so_check_2.mod"
        mod_path.write_text(dynare_mod.mod_text, encoding="utf-8")

        env = os.environ.copy()
        dynare_root = _find_dynare_root(dynare)
        if dynare_root is not None:
            env.setdefault("DYNARE_ROOT", dynare_root)

        proc = subprocess.run(
            [dynare, str(mod_path), "noclearall", "nolog", "nograph"],
            cwd=td_path,
            capture_output=True,
            text=True,
            timeout=240,
            env=env,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "Dynare run failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        results_candidates = sorted(td_path.rglob("dynare_so_check_2_results.mat"))
        assert results_candidates, f"Expected results file not found under: {td_path}"
        results_path = results_candidates[0]

        mat = loadmat(results_path, squeeze_me=True, struct_as_record=False)
        oo_ = mat["oo_"]
        M_ = mat["M_"]

        endo_names_decl = _matlab_char_array_to_strings(_get_struct_field(M_, "endo_names"))
        exo_names = _matlab_char_array_to_strings(_get_struct_field(M_, "exo_names"))

        order_var = np.asarray(_get_struct_field(oo_.dr, "order_var")).astype(int).ravel()
        if order_var.size:
            dyn_row_names = [endo_names_decl[i - 1] for i in order_var]
        else:
            dyn_row_names = list(endo_names_decl)

        state_var = np.asarray(_get_struct_field(oo_.dr, "state_var")).astype(int).ravel()
        state_names = [endo_names_decl[i - 1] for i in state_var]

        nendo = len(dyn_row_names)
        nstate = len(state_var)
        nexo = len(exo_names)

        ours_endo_idx = [ours["endo_names"].index(n) for n in dyn_row_names]

        def _state_base(n: str) -> str:
            return n[:-5] if n.endswith("_LAG1") else n

        ours_state_idx = []
        for dn in state_names:
            candidates = [i for i, sn in enumerate(ours["state_names"]) if _state_base(sn) == dn or sn == dn]
            if not candidates:
                raise AssertionError(f"Could not map Dynare state '{dn}' into our state_names={ours['state_names']}")
            ours_state_idx.append(candidates[0])

        ours_ghx = ours["ghx"][ours_endo_idx, :][:, ours_state_idx]
        ours_ghu = ours["ghu"][ours_endo_idx, :]
        ours_ghxx = ours["ghxx"][ours_endo_idx, :]
        ours_ghxu = ours["ghxu"][ours_endo_idx, :]
        ours_ghuu = ours["ghuu"][ours_endo_idx, :]
        ours_ghs2 = ours["ghs2"][ours_endo_idx, :]

        dyn_ghx = _ensure_2d(_get_struct_field(oo_.dr, "ghx"), nendo, nstate)
        dyn_ghu = _ensure_2d(_get_struct_field(oo_.dr, "ghu"), nendo, nexo)
        dyn_ghxx = _ensure_2d(_get_struct_field(oo_.dr, "ghxx"), nendo, nstate * nstate)
        dyn_ghxu = _ensure_2d(_get_struct_field(oo_.dr, "ghxu"), nendo, nstate * nexo)
        dyn_ghuu = _ensure_2d(_get_struct_field(oo_.dr, "ghuu"), nendo, nexo * nexo)
        dyn_ghs2 = _ensure_2d(_get_struct_field(oo_.dr, "ghs2"), nendo, 1)

        assert_allclose(ours_ghx, dyn_ghx, rtol=5e-6, atol=5e-8)
        assert_allclose(ours_ghu, dyn_ghu, rtol=5e-6, atol=5e-8)
        assert_allclose(ours_ghxx, dyn_ghxx, rtol=5e-5, atol=5e-7)
        assert_allclose(ours_ghxu, dyn_ghxu, rtol=5e-5, atol=5e-7)
        assert_allclose(ours_ghuu, dyn_ghuu, rtol=5e-5, atol=5e-7)
        assert_allclose(dyn_ghs2, ours_ghs2, rtol=5e-5, atol=5e-7)
