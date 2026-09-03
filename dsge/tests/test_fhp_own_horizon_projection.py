import io
from importlib.resources import files

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from dsge.parse_yaml import ValidationError, read_yaml


EXAMPLE = files("dsge") / "examples" / "fhp" / "nk_endogenous_own_horizon_projection.yaml"


def _row(model, lhs_name):
    equations = model["equations"]["cycle"]["plan"] + model["equations"]["static"]
    for i, equation in enumerate(equations):
        if getattr(equation.lhs, "name", None) == lhs_name:
            return i
    raise AssertionError(f"missing row {lhs_name!r}")


def _compile(fhp, *, household, pricing, belief_mode):
    spec = {"default": pricing, "by_lhs": {"y": household, "pi": pricing}}
    compiled = fhp.compile_model(k=spec, belief_mode=belief_mode)
    compiled.system_matrices(fhp.p0())
    return compiled


def test_common_horizon_projection_nests_standard_fhp():
    switching = read_yaml(str(EXAMPLE))
    fhp = switching.fhp_model

    for horizon in range(5):
        standard = _compile(
            fhp, household=horizon, pricing=horizon, belief_mode="correct"
        )
        projected = _compile(
            fhp,
            household=horizon,
            pricing=horizon,
            belief_mode="own_horizon_projection",
        )
        for name in ("A_cycle", "B_cycle", "A_trend", "B_trend"):
            assert_allclose(
                getattr(projected, name), getattr(standard, name), rtol=0, atol=1e-12
            )


def test_off_diagonal_rows_use_their_own_diagonal_continuations():
    switching = read_yaml(str(EXAMPLE))
    fhp = switching.fhp_model
    current = _compile(
        fhp, household=1, pricing=2, belief_mode="own_horizon_projection"
    )
    household_predecessor = _compile(
        fhp, household=0, pricing=0, belief_mode="correct"
    )
    pricing_predecessor = _compile(
        fhp, household=1, pricing=1, belief_mode="correct"
    )
    params = fhp.p0()
    P = current.P(params)

    y_row = _row(fhp, "y")
    pi_row = _row(fhp, "pi")
    predecessors = {
        y_row: household_predecessor,
        pi_row: pricing_predecessor,
    }

    alphaC = current.alphaC_cycle(params)
    alphaF = current.alphaF_cycle(params)
    alphaB = current.alphaB_cycle(params)
    betaS = current.betaS_cycle(params)
    for row, predecessor in predecessors.items():
        coefficient = alphaC[row, :] - alphaF[row, :] @ predecessor.A_cycle
        state_residual = coefficient @ current.A_cycle - alphaB[row, :]
        shock_residual = (
            coefficient @ current.B_cycle
            - alphaF[row, :] @ predecessor.B_cycle @ P
            - betaS[row, :]
        )
        assert_allclose(state_residual, 0.0, rtol=0, atol=1e-12)
        assert_allclose(shock_residual, 0.0, rtol=0, atol=1e-12)

    alphaC = current.alphaC_trend(params)
    alphaF = current.alphaF_trend(params)
    alphaB = current.alphaB_trend(params)
    for row, predecessor in predecessors.items():
        coefficient = alphaC[row, :] - alphaF[row, :] @ predecessor.A_trend
        state_residual = coefficient @ current.A_trend - alphaB[row, :]
        assert_allclose(state_residual, 0.0, rtol=0, atol=1e-12)

    correct_pair = _compile(fhp, household=1, pricing=2, belief_mode="correct")
    assert not np.allclose(current.B_cycle, correct_pair.B_cycle, rtol=0, atol=1e-10)


def test_endogenous_choice_uses_diagonal_perceived_economy():
    model = read_yaml(str(EXAMPLE))
    assert model.belief_mode == "own_horizon_projection"
    assert model.state_names == ["d"]
    state = np.array([0.25])
    params = model.p0

    for component, opponent in (("household", "pricing"), ("pricing", "household")):
        low = model.component_best_response(
            params,
            state,
            t=0,
            component=component,
            other_horizons={opponent: 0},
        )
        high = model.component_best_response(
            params,
            state,
            t=0,
            component=component,
            other_horizons={opponent: model.k_max[opponent]},
        )
        assert low == high

    chosen = model.choose_regime(params, state, t=0)
    diagnostics = model.simultaneous_regime_diagnostics(params, state, t=0)
    assert diagnostics.equilibria == (chosen,)
    mats = model.get_mats(params, chosen)
    assert all(np.isfinite(matrix).all() for matrix in mats)


def test_own_horizon_projection_rejects_aggregate_expectation_rows():
    data = yaml.safe_load(EXAMPLE.read_text())
    data["declarations"]["expectations"] = 1
    with pytest.raises(ValueError, match="different row owners"):
        read_yaml(io.StringIO(yaml.safe_dump(data, sort_keys=False)))


def test_schema_rejects_unknown_belief_mode():
    data = yaml.safe_load(EXAMPLE.read_text())
    data["declarations"]["stopping_rule"]["belief_mode"] = "wishful"
    with pytest.raises(ValidationError, match="belief_mode"):
        read_yaml(io.StringIO(yaml.safe_dump(data, sort_keys=False)))


def test_nk_projection_matches_matrix_geometric_closed_form():
    model = read_yaml(str(EXAMPLE))
    fhp = model.fhp_model
    values = dict(zip(model.parameter_names, model.p0))
    beta, sigma, kappa, rho = (values[name] for name in ("beta", "sigma", "kappa", "rho_d"))
    D = np.linalg.inv(np.array([[1.0, 0.0], [-kappa, 1.0]]))
    M_H = D @ np.array([[1.0, sigma], [0.0, 0.0]])
    M_F = D @ np.array([[0.0, 0.0], [0.0, beta]])
    M = M_H + M_F
    b = D @ np.array([sigma, 0.0])

    for household in range(4):
        for pricing in range(4):
            H = sum(
                (np.linalg.matrix_power(rho * M, j) for j in range(household)),
                start=np.zeros((2, 2)),
            )
            F = sum(
                (np.linalg.matrix_power(rho * M, j) for j in range(pricing)),
                start=np.zeros((2, 2)),
            )
            expected = b + rho * M_H @ H @ b + rho * M_F @ F @ b
            compiled = _compile(
                fhp,
                household=household,
                pricing=pricing,
                belief_mode="own_horizon_projection",
            )
            assert_allclose(compiled.B_cycle[:, 0], expected, rtol=0, atol=1e-12)


def test_projection_preserves_common_horizon_lags_and_terminal_values():
    path = files("dsge") / "examples" / "fhp" / "fhp.yaml"
    fhp = read_yaml(str(path))
    params = fhp.p0()
    for horizon in (0, 1, 4):
        standard = fhp.compile_model(k=horizon)
        projected = fhp.compile_model(k=horizon, belief_mode="own_horizon_projection")
        for expected, actual in zip(
            standard.system_matrices(params), projected.system_matrices(params)
        ):
            assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_projection_example_simulation_and_girf_are_finite():
    model = read_yaml(str(EXAMPLE))
    simulation = model.simulate(model.p0, T=20, seed=7)
    assert np.isfinite(simulation["y_path"]).all()
    assert np.isfinite(simulation["x_path"]).all()
    assert np.any(simulation["s_path"][:, 0] != simulation["s_path"][:, 1])

    girf = model.girf(model.p0, shock="e_d", h=4, reps=5, shock_size=0.1, seed=7)
    assert np.isfinite(girf["girf"].to_numpy()).all()


def test_candidate_projects_onto_rows_with_fixed_actual_horizons():
    data = yaml.safe_load(EXAMPLE.read_text())
    components = data["declarations"]["stopping_rule"]["components"]
    del components["pricing"]
    components["household"]["policy_object"] = "pi"
    model = read_yaml(io.StringIO(yaml.safe_dump(data, sort_keys=False)))
    params = model.p0
    state = np.array([1.0])
    info = model.info_func(state, 0, {})
    perceived_pi = model.policy_object(params, info, "household", 2, {})

    common = model.fhp_model.compile_model(k=2)
    common.system_matrices(model.fhp_model.p0())
    assert_allclose(perceived_pi, common.B_cycle[1, 0], rtol=0, atol=1e-12)

    # The actual unowned pricing row still has the fixed baseline horizon zero.
    _, _, ZZ, DD, _, _ = model.get_mats(params, (2,))
    actual_pi = (ZZ @ state + DD)[1]
    assert not np.isclose(actual_pi, perceived_pi, rtol=0, atol=1e-10)
