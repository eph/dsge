"""OBC wrappers preserve explicit host authority across every selected regime."""
from copy import deepcopy

import numpy as np
import pytest

from dsge.DSGE import DSGE
from dsge.obc import OccBinModel


@pytest.fixture
def external_obc(tmp_path):
    marker = tmp_path / "imported"
    plugin = tmp_path / "plugin.py"
    plugin.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef half_rho(x):\n    return x/2\n")
    data = {
        "declarations": {
            "name": "obc_external", "variables": ["x"], "parameters": ["rho"],
            "shocks": ["e"], "observables": ["x"], "auxiliary_parameters": ["gamma"],
            "external": {"file": str(plugin), "names": ["half_rho"]},
        },
        "equations": {"model": ["x = gamma*x(-1) + e"], "observables": {"x": "x"}},
        "calibration": {"parameters": {"rho": 0.8}, "auxiliary_parameters": {"gamma": "half_rho(rho)"}},
    }
    regimes = {name: DSGE.read(deepcopy(data)) for name in ("normal", "low", "high")}
    constraints = [
        {"name": "low", "when": "x < -1", "binding_regime": "low"},
        {"name": "high", "when": "x > 1", "binding_regime": "high"},
    ]
    return OccBinModel(regimes=regimes, constraint=constraints, yaml=data), marker


def test_obc_default_deny_and_registry_snapshot_reaches_additional_regimes(external_obc):
    model, marker = external_obc
    with pytest.raises(PermissionError):
        model.compile_model()
    registry = {"half_rho": lambda x: x/2}
    compiled = model.compile_model(external_functions=registry)
    registry.clear()
    result = compiled.simulate(model.p0(), np.array([[2.0], [0.0], [0.0]]))
    assert result["regimes"][0] == "high"
    np.testing.assert_allclose(result["states"][:, 0], [2.0, 0.8, 0.32])
    with pytest.raises(PermissionError):
        model.compile_model()
    assert not marker.exists()


@pytest.mark.parametrize("entrypoint", ["irf_obc", "simulate_obc", "impulse_response", "simulate", "impulse_response_states", "impulse_response_observables"])
def test_obc_convenience_calls_forward_host_registry(external_obc, entrypoint):
    model, marker = external_obc
    kwargs = {"external_functions": {"half_rho": lambda x: x/2}}
    if "simulate" in entrypoint:
        kwargs.update(eps_path=np.array([[0.5], [0.0]]), horizon=2)
    else:
        kwargs.update(h=2)
    assert getattr(model, entrypoint)(model.p0(), **kwargs) is not None
    assert not marker.exists()


def test_obc_explicit_file_authority_does_not_make_later_calls_trusted(external_obc):
    model, marker = external_obc
    model.compile_model(allow_external_code=True)
    assert marker.exists()
    with pytest.raises(PermissionError):
        model.compile_regime("normal")
