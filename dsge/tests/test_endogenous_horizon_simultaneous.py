import copy
import io
from importlib.resources import files

import numpy as np
import pytest
import yaml

from dsge.endogenous_horizon_switching import (
    EndogenousHorizonSwitchingModel,
    MultipleHorizonEquilibriaError,
    NoPureHorizonEquilibriumError,
)
from dsge.parse_yaml import ValidationError, read_yaml


def _make_binary_game(
    best_response_rule,
    *,
    components=("alpha", "beta"),
    selection_order=None,
    selection_mode="simultaneous",
    equilibrium_selection="error",
):
    mb_calls = []

    def solve_given_regime(params, regime):
        scale = 1.0 + 0.1 * sum(regime)
        return (
            np.array([[0.5]]),
            np.array([[1.0]]),
            np.array([[scale]]),
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([[0.25]]),
        )

    def info_func(x_t, t, chosen):
        return {"x": np.asarray(x_t), "t": int(t), "chosen": dict(chosen)}

    def mb_func(params, info_t, component, k_plus_1, chosen):
        assert int(k_plus_1) == 1
        assert info_t["chosen"] == dict(chosen)
        mb_calls.append((str(component), tuple(chosen.items())))
        desired = int(best_response_rule(str(component), dict(chosen)))
        return 2.0 if desired == 1 else 0.0

    kwargs = {}
    if selection_mode is not None:
        kwargs["selection_mode"] = selection_mode

    model = EndogenousHorizonSwitchingModel(
        components=list(components),
        k_max=1,
        cost_params=(1.0, 0.0),
        lam=1.0,
        solve_given_regime=solve_given_regime,
        policy_object=lambda params, info_t, component, k, chosen: 0.0,
        info_func=info_func,
        mb_func=mb_func,
        selection_order=selection_order,
        equilibrium_selection=equilibrium_selection,
        **kwargs,
    )
    model.shock_names = ["innovation"]
    model.obs_names = ["observable"]
    return model, mb_calls


def _choose_mapping(model):
    regime = model.choose_regime(np.array([0.0]), np.array([0.0]), t=0)
    return model.unpack_regime(regime)


def test_sequential_selection_remains_the_default_and_preserves_ordering():
    def sequential_rule(component, chosen):
        if component == "alpha":
            return 1
        return chosen.get("alpha", 0)

    alpha_first, _ = _make_binary_game(
        sequential_rule,
        selection_order=["alpha", "beta"],
        selection_mode=None,
    )
    beta_first, _ = _make_binary_game(
        sequential_rule,
        selection_order=["beta", "alpha"],
        selection_mode=None,
    )

    assert alpha_first.selection_mode == "sequential"
    assert _choose_mapping(alpha_first) == {"alpha": 1, "beta": 1}
    assert _choose_mapping(beta_first) == {"alpha": 1, "beta": 0}


def test_unique_simultaneous_equilibrium_and_best_response_cache():
    def unique_rule(component, others):
        if component == "alpha":
            return others["beta"]
        return 1

    model, mb_calls = _make_binary_game(unique_rule)

    assert model.component_best_response(
        np.array([0.0]),
        np.array([0.0]),
        t=0,
        component="alpha",
        other_horizons={"beta": 1},
    ) == 1

    mb_calls.clear()
    diagnostics = model.simultaneous_regime_diagnostics(
        np.array([0.0]), np.array([0.0]), t=0
    )
    assert diagnostics.n_profiles == 4
    assert diagnostics.equilibrium_mappings() == ({"alpha": 1, "beta": 1},)
    # Four distinct conditional responses, rather than 4 profiles x 2 components.
    assert len(mb_calls) == 4
    assert _choose_mapping(model) == {"alpha": 1, "beta": 1}


def test_simultaneous_selection_is_invariant_to_declaration_and_selection_order():
    def unique_rule(component, others):
        if component == "alpha":
            return others["beta"]
        return 1

    forward, _ = _make_binary_game(
        unique_rule,
        components=("alpha", "beta"),
        selection_order=("beta", "alpha"),
    )
    reversed_model, _ = _make_binary_game(
        unique_rule,
        components=("beta", "alpha"),
        selection_order=("alpha", "beta"),
    )

    assert _choose_mapping(forward) == {"alpha": 1, "beta": 1}
    assert _choose_mapping(reversed_model) == {"beta": 1, "alpha": 1}
    assert dict(_choose_mapping(forward)) == dict(_choose_mapping(reversed_model))


def test_multiple_equilibria_error_and_explicit_selection_policies():
    def coordination_rule(component, others):
        other = "beta" if component == "alpha" else "alpha"
        return others[other]

    model, _ = _make_binary_game(coordination_rule)
    with pytest.raises(MultipleHorizonEquilibriaError) as exc_info:
        _choose_mapping(model)

    error = exc_info.value
    assert error.diagnostics.equilibrium_mappings() == (
        {"alpha": 0, "beta": 0},
        {"alpha": 1, "beta": 1},
    )
    assert "alpha=0" in str(error)
    assert "alpha=1" in str(error)
    assert "lexicographic_min" in str(error)

    choose_low, _ = _make_binary_game(
        coordination_rule,
        equilibrium_selection="lexicographic_min",
    )
    choose_high, _ = _make_binary_game(
        coordination_rule,
        equilibrium_selection="lexicographic_max",
    )
    choose_high_reversed, _ = _make_binary_game(
        coordination_rule,
        components=("beta", "alpha"),
        selection_order=("alpha", "beta"),
        equilibrium_selection="lexicographic_max",
    )
    assert _choose_mapping(choose_low) == {"alpha": 0, "beta": 0}
    assert _choose_mapping(choose_high) == {"alpha": 1, "beta": 1}
    assert _choose_mapping(choose_high_reversed) == {"beta": 1, "alpha": 1}


def test_no_pure_equilibrium_reports_best_response_near_misses():
    def matching_pennies_rule(component, others):
        if component == "alpha":
            return others["beta"]
        return 1 - others["alpha"]

    model, _ = _make_binary_game(matching_pennies_rule)
    with pytest.raises(NoPureHorizonEquilibriumError) as exc_info:
        _choose_mapping(model)

    error = exc_info.value
    assert error.diagnostics.n_profiles == 4
    assert error.diagnostics.equilibria == ()
    assert "No pure simultaneous horizon equilibrium" in str(error)
    assert "deviating:" in str(error)
    assert "->" in str(error)


def test_parameter_dependent_cost_and_lambda_work_in_simultaneous_mode():
    def solve_given_regime(params, regime):
        return (
            np.array([[0.5]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([[0.25]]),
        )

    def info_func(x_t, t, chosen):
        return {"x": np.asarray(x_t), "chosen": dict(chosen)}

    def policy_object(params, info_t, component, k, chosen):
        if component == "alpha":
            amplitude = 2.0 if chosen["beta"] == 1 else 0.0
        else:
            amplitude = 2.0
        return float(k) * amplitude

    model = EndogenousHorizonSwitchingModel(
        components=["alpha", "beta"],
        k_max=1,
        cost_params=(1.0, 0.0),
        lam=1.0,
        cost_func=lambda params, component: (float(params[0]), 0.0),
        lam_func=lambda params, component: float(params[1]),
        solve_given_regime=solve_given_regime,
        policy_object=policy_object,
        info_func=info_func,
        selection_mode="simultaneous",
    )

    low_cost = model.unpack_regime(
        model.choose_regime(np.array([1.0, 1.0]), np.array([0.0]), t=0)
    )
    high_cost = model.unpack_regime(
        model.choose_regime(np.array([3.0, 1.0]), np.array([0.0]), t=0)
    )
    assert low_cost == {"alpha": 1, "beta": 1}
    assert high_cost == {"alpha": 0, "beta": 0}


def test_simulation_girf_and_particle_filter_use_simultaneous_selection():
    def unique_rule(component, others):
        if component == "alpha":
            return others["beta"]
        return 1

    model, _ = _make_binary_game(unique_rule)
    params = np.array([0.0])

    simulation = model.simulate(params, T=5, seed=7)
    assert simulation["s_path"].shape == (5, 2)
    assert np.all(simulation["s_path"] == np.array([1, 1]))

    girf = model.girf(params, shock="innovation", h=4, reps=3, seed=7)
    assert girf["girf"].shape == (4, 1)
    assert np.all(girf["k_base_mean"].to_numpy() == 1.0)
    assert np.all(girf["k_shocked_mean"].to_numpy() == 1.0)

    loglik, stats = model.pf_loglik(
        params,
        np.zeros((4, 1)),
        nparticles=12,
        seed=7,
    )
    assert np.isfinite(loglik)
    assert stats["k_mean"].shape == (4, 2)
    assert np.allclose(stats["k_mean"], 1.0)


def _switching_yaml_dict():
    return {
        "declarations": {
            "type": "switching_ssm",
            "name": "simultaneous_yaml_test",
            "components": ["beta", "alpha"],
            "states": ["x"],
            "shocks": ["innovation"],
            "observables": ["policy"],
            "parameters": [],
            "horizon_choice": {
                "selection_mode": "simultaneous",
                "equilibrium_selection": "lexicographic_max",
                "selection_order": ["alpha", "beta"],
                "components": {
                    "beta": {
                        "k_max": 1,
                        "cost": {"a": 0.1},
                        "lambda": 1.0,
                        "policy_object": "policy",
                    },
                    "alpha": {
                        "k_max": 1,
                        "cost": {"a": 0.1},
                        "lambda": 1.0,
                        "policy_object": "policy",
                    },
                },
            },
        },
        "model": {
            "TT": [[0.5]],
            "RR": [[1.0]],
            "ZZ": [[1.0]],
            "DD": ["k_alpha + k_beta"],
            "QQ": [[1.0]],
            "HH": [[0.25]],
        },
        "calibration": {"parameters": {}},
    }


def test_switching_yaml_round_trip_preserves_simultaneous_configuration():
    yaml_text = yaml.safe_dump(_switching_yaml_dict(), sort_keys=False)
    with pytest.warns(DeprecationWarning, match="switching_ssm"):
        model = read_yaml(io.StringIO(yaml_text))

    assert model.selection_mode == "simultaneous"
    assert model.equilibrium_selection == "lexicographic_max"
    assert model.selection_order == ["alpha", "beta"]
    regime = model.choose_regime(model.p0, np.array([0.0]), t=0)
    assert model.unpack_regime(regime) == {"beta": 1, "alpha": 1}


def test_two_component_fhp_yaml_compiles_and_simulates_in_simultaneous_mode():
    yaml_path = files("dsge") / "examples" / "fhp" / "fhp_endogenous_two_component.yaml"
    model = read_yaml(str(yaml_path))

    assert model.selection_mode == "simultaneous"
    assert model.equilibrium_selection == "error"
    x0 = np.zeros(len(model.state_names))
    diagnostics = model.simultaneous_regime_diagnostics(model.p0, x0, t=0)
    assert diagnostics.n_profiles == 55
    assert diagnostics.equilibrium_mappings() == ({"hh": 0, "pricing": 0},)

    simulation = model.simulate(model.p0, T=3, seed=123, x0=x0)
    assert simulation["s_path"].shape == (3, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_mode", "iterated"),
        ("equilibrium_selection", "first_declared"),
    ],
)
def test_switching_yaml_rejects_invalid_simultaneous_configuration(field, value):
    data = copy.deepcopy(_switching_yaml_dict())
    data["declarations"]["horizon_choice"][field] = value
    with pytest.raises(ValidationError, match=field):
        read_yaml(io.StringIO(yaml.safe_dump(data, sort_keys=False)))
