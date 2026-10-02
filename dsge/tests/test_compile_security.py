"""Trust and resource limits at the numerical compilation boundary."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sympy

from dsge import read_yaml
from dsge.symbols import Shock


def _model(external_file=None):
    declaration = ""
    auxiliary = ""
    coefficient = "rho"
    if external_file is not None:
        declaration = f"""
  auxiliary_parameters: [gamma]
  external:
    names: [half_rho]
    file: {str(external_file)!r}
"""
        auxiliary = "\n  auxiliary_parameters:\n    gamma: half_rho(rho)\n"
        coefficient = "gamma"
    return read_yaml(io.StringIO(f"""
declarations:
  name: compile_security
  variables: [x]
  parameters: [rho]
  shocks: [e]
  observables: [x]
  measurement_errors: [me_x]
{declaration}
equations:
  model:
    - x = {coefficient}*x(-1) + e
  observables:
    x: x
calibration:
  parameters:
    rho: 0.8
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 0.01
{auxiliary}
"""))


@pytest.fixture
def external_bundle(tmp_path):
    marker = tmp_path / "imported.txt"
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "def half_rho(rho):\n    return rho / 2\n"
    )
    return _model(plugin), marker


@pytest.mark.parametrize("entrypoint", ["linear", "matrices", "solve", "order2"])
def test_external_files_default_deny_before_execution(external_bundle, entrypoint):
    model, marker = external_bundle
    # Model metadata cannot grant itself the host's capability.
    model["__data__"]["declarations"]["external"]["allow_external_code"] = True
    with pytest.raises(PermissionError, match="external_functions"):
        if entrypoint == "linear":
            model.compile_model()
        elif entrypoint == "matrices":
            model.python_sims_matrices()
        elif entrypoint == "solve":
            model.solve_second_order()
        else:
            model.compile_model(order=2)
    assert not marker.exists()


@pytest.mark.parametrize("order", [1, 2])
def test_explicit_file_opt_in_is_scoped_to_each_request(external_bundle, order):
    model, marker = external_bundle
    if order == 1:
        compiled = model.compile_model(allow_external_code=True)
        assert -compiled.GAM1(model.p0())[0, 0] == pytest.approx(0.4)
        with pytest.raises(PermissionError):
            model.compile_model()
    else:
        solution = model.solve_second_order(allow_external_code=True)
        assert solution.hx[0, 0] == pytest.approx(0.4)
        with pytest.raises(PermissionError):
            model.solve_second_order()
    assert marker.read_text() == "imported"


@pytest.mark.parametrize("order", [1, 2])
def test_registry_never_imports_file_and_cache_tracks_callables(external_bundle, order):
    model, marker = external_bundle
    # Keep both registries alive and revisit the first after switching contexts.
    registries = [{"half_rho": lambda value: value / 2}, {"half_rho": lambda value: value / 4}]
    for registry, expected in [(registries[0], 0.4), (registries[1], 0.2), (registries[0], 0.4)]:
        if order == 1:
            compiled = model.compile_model(external_functions=registry)
            actual = -compiled.GAM1(model.p0())[0, 0]
        else:
            actual = model.solve_second_order(external_functions=registry).hx[0, 0]
        assert actual == pytest.approx(expected)
    assert not marker.exists()


def test_failed_registry_recompile_preserves_complete_previous_cache(external_bundle):
    model, marker = external_bundle
    registry_a = {"half_rho": lambda value: value / 2}
    registry_b = {"half_rho": lambda value: value / 4}
    first = model.compile_model(external_functions=registry_a)
    previous_auxiliary = model.lambdify_auxiliary
    original = model.lambdify
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("interrupted numerical compilation")
        return original(*args, **kwargs)

    with patch.object(model, "lambdify", side_effect=interrupted):
        with pytest.raises(ValueError, match="interrupted"):
            model.compile_model(external_functions=registry_b)
    assert model.lambdify_auxiliary is previous_auxiliary
    assert model.GAM1 is first.GAM1
    retry = model.compile_model(external_functions=registry_a)
    assert -retry.GAM1(model.p0())[0, 0] == pytest.approx(0.4)
    changed = model.compile_model(external_functions=registry_b)
    assert -changed.GAM1(model.p0())[0, 0] == pytest.approx(0.2)
    assert not marker.exists()


@pytest.mark.parametrize("registry", [{}, {"half_rho": 2}])
def test_registry_must_supply_callable_for_each_declared_function(external_bundle, registry):
    model, marker = external_bundle
    with pytest.raises(ValueError, match="callable"):
        model.compile_model(external_functions=registry)
    assert not marker.exists()


def test_order2_wrapper_carries_registry_to_policy_and_measurement(external_bundle):
    model, marker = external_bundle
    registry = {"half_rho": lambda value: value / 2}
    compiled = model.compile_model(order=2, external_functions=registry)
    # The wrapper owns a snapshot of caller-supplied capabilities.
    registry.clear()
    assert np.isfinite(compiled.log_lik(model.p0(), y=np.zeros((3, 1)), nparticles=20, seed=1))
    assert compiled.simulate(model.p0(), nsim=3, seed=1).shape == (3, 1)
    assert not marker.exists()


def test_order2_wrapper_carries_explicit_file_opt_in(external_bundle):
    model, marker = external_bundle
    compiled = model.compile_model(order=2, allow_external_code=True)
    assert not marker.exists()  # Compiling the wrapper is still lazy.
    assert compiled.simulate(model.p0(), nsim=3, seed=1).shape == (3, 1)
    assert marker.read_text() == "imported"
    with pytest.raises(PermissionError):
        model.compile_model(order=2)


def test_tensor_limit_runs_before_even_opted_in_external_import(external_bundle):
    model, marker = external_bundle
    with pytest.raises(ValueError, match="50 entries"):
        model.solve_second_order(max_tensor_entries=49, allow_external_code=True)
    assert not marker.exists()


def test_core_reduction_preserves_host_registry(external_bundle):
    model, marker = external_bundle
    compiled = model.compile_model(
        lre_reduction="core", external_functions={"half_rho": lambda value: value / 2}
    )
    assert -compiled.GAM1(model.p0())[0, 0] == pytest.approx(0.4)
    assert not marker.exists()


def _block_symbolic_compilation(monkeypatch, model):
    blocked = Mock(side_effect=AssertionError("symbolic work occurred before the guard"))
    monkeypatch.setattr(sympy, "hessian", blocked)
    monkeypatch.setattr(sympy.matrices.matrixbase.MatrixBase, "jacobian", blocked)
    monkeypatch.setattr(model, "lambdify", blocked)
    return blocked


@pytest.mark.parametrize("entrypoint", ["solve", "compile"])
def test_tensor_limit_rejects_before_symbolic_work(monkeypatch, entrypoint):
    model = _model()
    blocked = _block_symbolic_compilation(monkeypatch, model)
    # One endogenous plus one lag auxiliary: n=2, nv=5, n*nv*nv=50.
    with pytest.raises(ValueError, match="50 entries"):
        if entrypoint == "solve":
            model.solve_second_order(max_tensor_entries=49)
        else:
            model.compile_model(order=2, max_tensor_entries=49)
    blocked.assert_not_called()
    assert getattr(model, "_second_order_compiled", None) is None


def test_tensor_limit_rechecks_cached_compilation(monkeypatch):
    model = _model()
    model.solve_second_order(max_tensor_entries=50)
    cached = model._second_order_compiled
    blocked = _block_symbolic_compilation(monkeypatch, model)
    with pytest.raises(ValueError, match="50 entries"):
        model.solve_second_order(max_tensor_entries=49)
    assert model._second_order_compiled is cached
    blocked.assert_not_called()


@pytest.mark.parametrize("operation", ["likelihood", "simulate"])
def test_order2_wrapper_enforces_budget_in_later_workflows(monkeypatch, operation):
    model = _model()
    compiled = model.compile_model(order=2, max_tensor_entries=50)
    # A model enlarged after wrapper construction must still respect its budget.
    model["shk_ordering"].append(Shock("extra"))
    blocked = _block_symbolic_compilation(monkeypatch, model)
    with pytest.raises(ValueError, match="72 entries"):
        if operation == "likelihood":
            compiled.log_lik(model.p0(), y=np.zeros((3, 1)), nparticles=20)
        else:
            compiled.simulate(model.p0(), nsim=3)
    blocked.assert_not_called()


@pytest.mark.parametrize("limit", [-1, True, float("nan"), float("inf")])
def test_invalid_tensor_budgets_cannot_bypass_limit(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        _model().solve_second_order(max_tensor_entries=limit)
