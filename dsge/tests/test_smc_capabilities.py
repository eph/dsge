"""Estimation wrappers preserve explicit compilation and YAML capabilities."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from dsge import read_yaml
from dsge.smc import smc_estimate_from_yaml, smc_submit_slurm


@pytest.fixture
def source_bundle(tmp_path):
    marker = tmp_path / "imported.txt"
    plugin = tmp_path / "helper.py"
    plugin.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "def half_rho(value):\n    return value / 2\n"
    )
    data = {
        "declarations": {
            "name": "smc_capabilities", "variables": ["x"], "shocks": ["e"],
            "parameters": ["rho"], "auxiliary_parameters": ["gamma"],
            "external": {"names": ["half_rho"], "file": str(plugin)},
        },
        "equations": {"model": ["x = gamma*x(-1) + e"], "observables": {"x": "x"}},
        "calibration": {
            "parameters": {"rho": 0.8}, "auxiliary_parameters": {"gamma": "half_rho(rho)"},
            "covariance": {"e": 0.01},
        },
    }
    source = tmp_path / "model.yaml"
    source.write_text(yaml.safe_dump(data))
    return source, marker


@pytest.mark.parametrize("entrypoint", ["local", "yaml"])
def test_smc_wrappers_deny_external_code_by_default(source_bundle, entrypoint):
    source, marker = source_bundle
    with patch("dsge.smc.smc_estimate") as estimator:
        with pytest.raises(PermissionError, match="disabled"):
            if entrypoint == "local":
                read_yaml(str(source)).estimate_smc()
            else:
                smc_estimate_from_yaml(str(source))
        estimator.assert_not_called()
    assert not marker.exists()


@pytest.mark.parametrize("entrypoint", ["local", "yaml"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("authority", ["registry", "file"])
def test_smc_wrappers_forward_trust_and_budget(source_bundle, entrypoint, order, authority):
    source, marker = source_bundle
    model = read_yaml(str(source))
    options = {"order": order, "max_tensor_entries": 50}
    if authority == "registry":
        options["external_functions"] = {"half_rho": lambda value: value / 2}
    else:
        options["allow_external_code"] = True
    with patch("dsge.smc.smc_estimate", return_value="sampled") as estimator:
        if entrypoint == "local":
            result = model.estimate_smc(**options)
        else:
            result = smc_estimate_from_yaml(str(source), **options)
    assert result == "sampled"
    compiled = estimator.call_args.args[0]
    if order == 1:
        coefficient = -compiled.GAM1(model.p0())[0, 0]
    else:
        assert compiled._max_tensor_entries == 50
        coefficient = compiled._policy(model.p0()).hx[0, 0]
    assert coefficient == pytest.approx(0.4)
    assert marker.exists() is (authority == "file")


@pytest.mark.parametrize("entrypoint", ["local", "yaml"])
def test_smc_wrappers_reject_oversized_order2_before_sampling(source_bundle, entrypoint):
    source, marker = source_bundle
    options = {"order": 2, "max_tensor_entries": 49, "allow_external_code": True}
    with patch("dsge.smc.smc_estimate") as estimator:
        with pytest.raises(ValueError, match="50 entries"):
            if entrypoint == "local":
                read_yaml(str(source)).estimate_smc(**options)
            else:
                smc_estimate_from_yaml(str(source), **options)
        estimator.assert_not_called()
    assert not marker.exists()


def test_smc_yaml_reload_requires_explicit_include_root(source_bundle):
    source, marker = source_bundle
    entry = source.with_name("entry.yaml")
    entry.write_text("!include model.yaml")
    registry = {"half_rho": lambda value: value / 2}
    with pytest.raises(ValueError, match="disabled"):
        smc_estimate_from_yaml(str(entry), external_functions=registry)
    with patch("dsge.smc.smc_estimate", return_value="sampled"):
        assert smc_estimate_from_yaml(
            str(entry), external_functions=registry,
            read_yaml_options={"include_root": str(source.parent), "max_include_depth": 1},
        ) == "sampled"
    assert not marker.exists()


@pytest.mark.parametrize("via_model", [False, True])
def test_slurm_submission_snapshots_explicit_options_without_running_jobs(source_bundle, via_model):
    source, marker = source_bundle
    registry = {"half_rho": lambda value: value / 2}
    yaml_options = {"include_root": str(source.parent), "max_include_bytes": 4096}
    executor = Mock()
    executor.submit.return_value = "queued"
    auto_executor = Mock(return_value=executor)
    options = {
        "external_functions": registry, "max_tensor_entries": 50,
        "order": 2, "read_yaml_options": yaml_options,
    }
    with patch.dict("sys.modules", {"submitit": SimpleNamespace(AutoExecutor=auto_executor)}):
        if via_model:
            assert read_yaml(str(source)).estimate_smc(backend="slurm", **options) == "queued"
        else:
            assert smc_submit_slurm(str(source), **options) == "queued"
    function, submitted_path = executor.submit.call_args.args
    submitted = executor.submit.call_args.kwargs
    assert function is smc_estimate_from_yaml
    assert submitted_path == str(source)
    assert submitted["external_functions"] == registry
    assert submitted["external_functions"] is not registry
    assert submitted["allow_external_code"] is False
    assert submitted["max_tensor_entries"] == 50
    assert submitted["read_yaml_options"] == yaml_options
    assert submitted["read_yaml_options"] is not yaml_options
    registry.clear()
    yaml_options.clear()
    assert callable(submitted["external_functions"]["half_rho"])
    assert "include_root" in submitted["read_yaml_options"]
    assert not marker.exists()


def test_slurm_serialization_failure_does_not_fall_back_to_file_authority(source_bundle):
    source, marker = source_bundle
    executor = Mock()
    executor.submit.side_effect = TypeError("registry cannot be serialized")
    with patch.dict("sys.modules", {"submitit": SimpleNamespace(AutoExecutor=Mock(return_value=executor))}):
        with pytest.raises(TypeError, match="cannot be serialized"):
            smc_submit_slurm(str(source), external_functions={"half_rho": lambda x: x / 2})
    assert executor.submit.call_count == 1
    assert executor.submit.call_args.kwargs["allow_external_code"] is False
    assert not marker.exists()


def test_yaml_read_options_are_narrow_host_options(source_bundle):
    source, _ = source_bundle
    with pytest.raises(ValueError, match="Unknown read_yaml_options"):
        smc_estimate_from_yaml(str(source), read_yaml_options={"allow_external_code": True})
    with pytest.raises(ValueError, match="Slurm"):
        read_yaml(str(source)).estimate_smc(read_yaml_options={"include_root": str(source.parent)})
