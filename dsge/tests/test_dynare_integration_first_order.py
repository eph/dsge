import io
import os

import pytest
from numpy.testing import assert_allclose

from dsge.parse_yaml import read_yaml


def _dynare_cmd() -> str | None:
    import shutil

    return shutil.which("dynare") or shutil.which("dynare-octave")


def _state_base(name: str) -> str:
    return name[:-5] if name.endswith("_LAG1") else name


def _compare_first_order_against_dynare(yaml_text: str):
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()

    # For linear models the first-order slices of the order-2 exporter are the same
    # decision-rule objects Dynare reports at order 1, but already arranged in
    # Dynare-like row/column layout.
    ours = m.solve_second_order(p0).as_dynare_like()
    dyn = m.dynare_first_order_solution(p0=p0)

    ours_endo_idx = [ours["endo_names"].index(name) for name in dyn.dyn_row_names]
    ours_state_idx = []
    for dyn_state in dyn.state_names:
        candidates = [
            i for i, state_name in enumerate(ours["state_names"])
            if _state_base(state_name) == dyn_state or state_name == dyn_state
        ]
        if not candidates:
            raise AssertionError(f"Could not map Dynare state '{dyn_state}' into {ours['state_names']}")
        ours_state_idx.append(candidates[0])

    assert_allclose(ours["ghx"][ours_endo_idx, :][:, ours_state_idx], dyn.ghx, rtol=5e-6, atol=5e-8)
    assert_allclose(ours["ghu"][ours_endo_idx, :], dyn.ghu, rtol=5e-6, atol=5e-8)


@pytest.mark.skipif(_dynare_cmd() is None, reason="Dynare not installed (expected `dynare` on PATH).")
@pytest.mark.skipif(os.environ.get("DSGE_RUN_DYNARE") != "1", reason="Set DSGE_RUN_DYNARE=1 to run Dynare checks.")
@pytest.mark.parametrize(
    "yaml_text",
    [
        """
declarations:
  name: dynare_fo_check_1
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
  covariance:
    e: 0.01
""",
        """
declarations:
  name: dynare_fo_check_2
  variables: [x, y, z]
  shocks: [e1, e2]
  parameters: [rho_x, rho_z, beta, psi]

equations:
  model:
    - x = rho_x*x(-1) + e1
    - z = rho_z*z(-1) + e2
    - y = beta*y(1) + x + psi*z
  observables:
    x: x
    y: y
    z: z

calibration:
  parameters:
    rho_x: 0.8
    rho_z: 0.6
    beta: 0.99
    psi: 0.4
  covariance:
    e1: 0.01
    e2: 0.02
""",
    ],
)
def test_first_order_matches_dynare(yaml_text):
    _compare_first_order_against_dynare(yaml_text)
