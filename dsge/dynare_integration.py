from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


def dynare_cmd() -> str | None:
    """Return a Dynare wrapper command if installed on PATH."""
    return shutil.which("dynare") or shutil.which("dynare-octave")


def find_dynare_root(dynare: str) -> str | None:
    """
    Best-effort guess of Dynare root directory (contains `matlab/dynare.m`).

    This mirrors the logic used in tests so that end-users don't have to set
    `DYNARE_ROOT` manually in common setups.
    """
    env_root = os.environ.get("DYNARE_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "matlab" / "dynare.m").exists():
            return str(p)

    cmd_path = Path(dynare)
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


@dataclass(frozen=True)
class DynareFirstOrderSolution:
    endo_names_decl: List[str]
    exo_names: List[str]
    dyn_row_names: List[str]
    state_names: List[str]
    ghx: np.ndarray
    ghu: np.ndarray
    eigval: np.ndarray


def run_dynare_mod_text(
    *,
    mod_text: str,
    model_name: str,
    timeout: int = 240,
    dynare: str | None = None,
    dynare_args: Tuple[str, ...] = ("noclearall", "nolog", "nograph"),
) -> Path:
    """
    Run Dynare on a `.mod` text and return the path to the `*_results.mat` file.
    """
    if dynare is None:
        dynare = dynare_cmd()
    if dynare is None:
        raise RuntimeError("Dynare not installed (expected `dynare` or `dynare-octave` on PATH).")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod_path = td_path / f"{model_name}.mod"
        mod_path.write_text(mod_text, encoding="utf-8")

        env = os.environ.copy()
        root = find_dynare_root(dynare)
        if root is not None:
            env.setdefault("DYNARE_ROOT", root)

        proc = subprocess.run(
            [dynare, str(mod_path), *dynare_args],
            cwd=td_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Dynare run failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        results_candidates = sorted(td_path.rglob(f"{model_name}_results.mat"))
        if not results_candidates:
            raise RuntimeError(f"Dynare completed but results file not found under: {td_path}")

        # Copy results out of the tempdir so the caller can load later.
        out_path = Path(tempfile.mkdtemp(prefix="dsge_dynare_results_", dir="/tmp")) / results_candidates[0].name
        out_path.write_bytes(results_candidates[0].read_bytes())
        return out_path


def load_first_order_solution(results_mat_path: str | Path) -> DynareFirstOrderSolution:
    """
    Load a Dynare `*_results.mat` file and extract first-order solution objects.
    """
    p = Path(results_mat_path)
    mat: Dict[str, Any] = loadmat(p, squeeze_me=True, struct_as_record=False)
    oo_ = mat["oo_"]
    M_ = mat["M_"]

    endo_names_decl = _matlab_char_array_to_strings(_get_struct_field(M_, "endo_names"))
    exo_names = _matlab_char_array_to_strings(_get_struct_field(M_, "exo_names"))

    order_var = np.asarray(_get_struct_field(oo_.dr, "order_var")).astype(int).ravel()
    if order_var.size:
        dyn_row_names = [endo_names_decl[i - 1] for i in order_var]  # 1-based -> 0-based
    else:
        dyn_row_names = list(endo_names_decl)

    state_var = np.asarray(_get_struct_field(oo_.dr, "state_var")).astype(int).ravel()
    state_names = [endo_names_decl[i - 1] for i in state_var]

    nendo = len(dyn_row_names)
    nstate = len(state_var)
    nexo = len(exo_names)

    ghx = _ensure_2d(_get_struct_field(oo_.dr, "ghx"), nendo, nstate)
    ghu = _ensure_2d(_get_struct_field(oo_.dr, "ghu"), nendo, nexo)
    eigval = np.asarray(_get_struct_field(oo_.dr, "eigval"), dtype=complex).ravel()

    return DynareFirstOrderSolution(
        endo_names_decl=endo_names_decl,
        exo_names=exo_names,
        dyn_row_names=dyn_row_names,
        state_names=state_names,
        ghx=ghx,
        ghu=ghu,
        eigval=eigval,
    )

