"""Adversarial data must never become statements in generated source."""

import io
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sympy

from dsge.codegen_safety import fortran_string
from dsge.dynare_export import to_dynare_mod, validate_generated_dynare
from dsge.dynare_integration import run_dynare_mod_text
from dsge.parse_yaml import read_yaml
from dsge.safe_expression import parse_expression
from dsge.symbols import Equation, Parameter, Shock, Variable
from dsge.translate import (
    generate_hardcoded_data_fortran, make_fortran_model, smc,
    translate, translate_fortran,
)


@pytest.fixture
def model():
    return read_yaml(io.StringIO("""
declarations:
  name: export_security
  variables: [x]
  shocks: [e, u]
  parameters: [rho, sigma]
equations:
  model: [x = rho*x(-1) + e + u]
  observables: {x: x}
calibration:
  parameters: {rho: 0.9, sigma: 0.1}
  covariance: {e: 0.01, u: 0.02}
"""))


@pytest.fixture
def external_model(tmp_path):
    # A benign host-owned fixture for testing explicit file-plugin authority.
    plugin = tmp_path / "helper.py"
    plugin.write_text("def half_rho(value):\n    return value / 2\n", encoding="utf-8")
    return read_yaml(io.StringIO(f"""
declarations:
  name: external_export
  variables: [x]
  shocks: [e]
  parameters: [rho]
  auxiliary_parameters: [gamma]
  external:
    names: [half_rho]
    file: {plugin}
equations:
  model: [x = gamma*x(-1) + e]
  observables: {{x: x}}
calibration:
  parameters: {{rho: 0.9}}
  auxiliary_parameters: {{gamma: half_rho(rho)}}
  covariance: {{e: 0.01}}
"""))


@pytest.mark.parametrize("entrypoint", [smc, make_fortran_model, translate_fortran, translate])
def test_fortran_wrappers_deny_external_files_by_default(external_model, tmp_path, entrypoint):
    args = (external_model, tmp_path / "output") if entrypoint in (translate_fortran, translate) else (external_model,)
    with patch("dsge.external_functions.SourceFileLoader.exec_module") as load:
        with pytest.raises(PermissionError, match="disabled"):
            entrypoint(*args)
        load.assert_not_called()
    assert not list(tmp_path.rglob("*.f90"))


@pytest.mark.parametrize("authority", ["registry", "file"])
def test_fortran_smc_forwards_explicit_external_authority(external_model, authority):
    registry = {"half_rho": lambda value: value / 2}
    kwargs = {"external_functions": registry} if authority == "registry" else {"allow_external_code": True}
    if authority == "registry":
        # Host functions take precedence over a model-selected path.
        external_model["__data__"]["declarations"]["external"]["file"] = "/unused/model/file.py"
    source = smc(external_model, **kwargs)
    assert "half_rho(para(1))" in source
    # Compiling once with authority never grants authority to a later call.
    with pytest.raises(PermissionError, match="disabled"):
        smc(external_model)


@pytest.mark.parametrize("authority", ["registry", "file"])
def test_fortress_wrapper_keeps_host_authority_out_of_build_options(external_model, authority, tmp_path):
    registry = {"half_rho": lambda value: value / 2}
    kwargs = {"external_functions": registry} if authority == "registry" else {"allow_external_code": True}
    builder = Mock(return_value="project")
    with patch.dict("sys.modules", {"fortress": SimpleNamespace(make_smc=builder)}):
        with patch("dsge.translate.write_prior_file"):
            with patch.object(external_model, "compile_model", wraps=external_model.compile_model) as compile_model:
                assert make_fortran_model(external_model, output_directory=str(tmp_path), **kwargs) == "project"
    assert len(compile_model.call_args_list) == 2
    for call in compile_model.call_args_list:
        assert call.kwargs["external_functions"] is (registry if authority == "registry" else None)
        assert call.kwargs["allow_external_code"] is (authority == "file")
    assert "half_rho(para(1))" in builder.call_args.args[0]
    assert "external_functions" not in builder.call_args.kwargs
    assert "allow_external_code" not in builder.call_args.kwargs


def test_legacy_fortran_dispatch_forwards_host_authority(external_model, tmp_path):
    registry = {"half_rho": lambda value: value / 2}
    # The legacy writer has optional dependencies; exercise real compilation,
    # but replace its file printers and template loop with inert mocks.
    with patch("dsge.translate.fortran_files", {}):
        with patch("dsge.translate.write_prior_file"), patch("dsge.translate.write_trans_file"), patch("dsge.translate.write_model_file"):
            with patch.object(external_model, "compile_model", wraps=external_model.compile_model) as compile_model:
                translate(external_model, tmp_path, external_functions=registry)
    compile_model.assert_called_once_with(external_functions=registry, allow_external_code=False)


@pytest.mark.parametrize("name", ["bad\nsteady;", "bad\rmodel;", "bad\x00", "bad\u2028steady;", "bad\t"])
def test_control_characters_in_name_fail_before_codegen(model, name):
    model["name"] = name
    with pytest.raises(ValueError, match="control characters"):
        to_dynare_mod(model)
    with patch.object(type(model), "compile_model") as compile_model:
        with pytest.raises(ValueError, match="control characters"):
            smc(model)
        compile_model.assert_not_called()


def test_apostrophes_and_comment_markers_are_inert(model):
    model["name"] = "O'Brien; ! // @# harmless"
    dynare = to_dynare_mod(model).mod_text
    assert dynare.splitlines()[0].endswith(model["name"])
    fortran = smc(model)
    assert "name = 'O''Brien; ! // @# harmless'" in fortran


@pytest.mark.parametrize("source", [
    "0; steady;", "0\nnative; disp(123); end;", "@#include 'unused'",
    "1 // comment", "1 /* comment */", "unknown(1)", "'text'",
])
@pytest.mark.parametrize("field", ["calibration", "covariance"])
def test_dynare_expression_text_is_not_source(model, source, field):
    if field == "calibration":
        model["calibration"]["rho"] = source
    else:
        model["covariance"] = {"e": source, "u": 0.02}
    with pytest.raises(ValueError):
        to_dynare_mod(model)


@pytest.mark.parametrize("field, symbol_type", [
    ("var_ordering", Variable), ("shk_ordering", Shock), ("parameters", Parameter),
])
@pytest.mark.parametrize("name", ["x; steady;", "x\nend;", "x/*comment*/", "α", "_x"])
def test_dynare_declaration_names_are_validated(model, field, symbol_type, name):
    model[field] = [symbol_type(name)]
    with pytest.raises(ValueError, match="identifier"):
        to_dynare_mod(model)


@pytest.mark.parametrize("expression", [
    sympy.Function("unknown")(1),
    sympy.Symbol("x; steady;"),
    sympy.Integral(sympy.Symbol("x"), sympy.Symbol("x")),
    sympy.oo,
])
def test_dynare_rejects_unsupported_symbolic_nodes(model, expression):
    model["perturb_eq"] = [Equation(Variable("x"), expression)]
    with pytest.raises(ValueError):
        to_dynare_mod(model)


@pytest.mark.parametrize("key", ["e,u; steady;", "e,u,x", "absent", "e\nend;"])
def test_dynare_covariance_keys_are_validated(model, key):
    model["covariance"] = {key: 0.01}
    with pytest.raises(ValueError):
        to_dynare_mod(model)


def test_dynare_preserves_math_lags_shocks_and_calibration(model):
    model["calibration"]["rho"] = "1 - sigma**2"
    model["covariance"] = {"e": "sigma**2", "u": 0.02, "e,u": "sigma/100"}
    model["perturb_eq"] = [Equation(Variable("x"),
        Parameter("rho") * Variable("x", date=-1) + Shock("e") +
        sympy.exp(Variable("x", date=1)) - 1)]
    text = to_dynare_mod(model, order=1).mod_text
    assert "x(-1)" in text and "x(1)" in text
    assert "exp(x(1))" in text
    assert "var e, u =" in text
    assert "rho = (1 + (-1 * ((sigma)^(2))));" in text
    assert "stoch_simul(order=1, irf=0);" in text


@pytest.mark.parametrize("source", [
    "(-1)**rho", "(-2)**rho", "(-rho)**sigma", "rho**(-sigma)",
    "rho**(-2)", "rho**(-1/2)", "(-rho - sigma)**(-rho + sigma)",
    "(rho**sigma)**rho", "rho**(sigma**rho)",
])
def test_dynare_power_serialization_preserves_operand_precedence(model, source):
    rho, sigma = Parameter("rho"), Parameter("sigma")
    context = {"rho": rho, "sigma": sigma}
    expression = parse_expression(source, context)
    model["perturb_eq"] = [Equation(Variable("x"), expression)]
    text = to_dynare_mod(model).mod_text
    equation = next(line for line in text.splitlines() if line.startswith("  x = "))
    printed = equation.removeprefix("  x = ").removesuffix(";")
    recovered = parse_expression(printed, context)
    assert recovered == expression
    assert recovered.subs({rho: 2, sigma: 3}) == expression.subs({rho: 2, sigma: 3})


@pytest.mark.parametrize("text", ["native; disp(123); end;", "@#include 'unused'", "steady;", "model; x = 0; end;"])
def test_generated_dynare_validator_rejects_other_programs(text):
    with pytest.raises(ValueError, match="generated Dynare"):
        validate_generated_dynare(text)


def test_automatic_dynare_rejects_injection_before_launch(model):
    model["calibration"]["rho"] = "0; native; disp(123); end;"
    with patch("dsge.dynare_integration.subprocess.run") as run:
        with pytest.raises(ValueError):
            model.dynare_first_order_solution(p0=[0.9, 0.1])
        run.assert_not_called()


def test_automatic_dynare_filename_is_independent_of_metadata(model, tmp_path):
    model["name"] = "42 O'Brien model α"
    results = tmp_path / "dsge_model_results.mat"
    with patch("dsge.dynare_integration.run_dynare_mod_text", return_value=results) as run:
        with patch("dsge.dynare_integration.load_first_order_solution", return_value="solution"):
            assert model.dynare_first_order_solution(p0=[0.9, 0.1]) == "solution"
    assert run.call_args.kwargs["model_name"] == "dsge_model"


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", "x\n", "x;", ""])
def test_dynare_runner_rejects_unsafe_filenames(name):
    with patch("dsge.dynare_integration.subprocess.run") as run:
        with pytest.raises(ValueError, match="identifier"):
            run_dynare_mod_text(mod_text="", model_name=name, dynare="dynare")
        run.assert_not_called()


def test_fortran_data_is_numeric_and_nan_is_explicit():
    text = generate_hardcoded_data_fortran(np.array([[1.0], [np.nan]]))
    assert "1.0_wp" in text
    assert "ieee_value(0.0_wp, ieee_quiet_nan)" in text
    with pytest.raises(ValueError):
        generate_hardcoded_data_fortran(np.array([["0; stop"]]))
    with pytest.raises(ValueError, match="finite"):
        generate_hardcoded_data_fortran(np.array([[np.inf]]))


@pytest.mark.parametrize("data", [
    np.full((2, 7), np.nan),
    np.array([[np.finfo(float).max, -np.finfo(float).max, np.finfo(float).tiny,
               np.nextafter(0.0, 1.0), -0.0, 1e20, -1e-20]]),
])
def test_fortran_data_respects_line_limits_and_compiles(data, tmp_path):
    generated = generate_hardcoded_data_fortran(data)
    assert max(map(len, generated.splitlines())) <= 132
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("Fortran compiler unavailable")
    rows, columns = data.shape
    source = tmp_path / "data.f90"
    source.write_text(f"""program generated_data
  use, intrinsic :: iso_fortran_env, only: wp => real64
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  type test_model
    real(wp) :: yy({columns}, {rows})
    logical :: MISSING_DATA = .false.
  end type
  type(test_model) :: self
{generated}
end program generated_data
""", encoding="utf-8")
    # Check syntax and numeric constants only; never execute generated programs.
    result = subprocess.run([compiler, "-Werror=line-truncation", "-fsyntax-only", str(source)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_fortran_legacy_module_name_is_validated_before_writing(model, tmp_path):
    model["name"] = "x; stop"
    with pytest.raises(ValueError, match="identifier"):
        translate_fortran(model, tmp_path)
    assert not list(tmp_path.iterdir())


def test_fortran_parameter_names_are_validated_before_compiling(model):
    model["parameters"] = [Parameter("rho; stop")]
    with patch.object(type(model), "compile_model") as compile_model:
        with pytest.raises(ValueError, match="identifier"):
            smc(model)
        compile_model.assert_not_called()


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="Fortran compiler unavailable")
@pytest.mark.parametrize("flags", [[], ["-fbackslash"]])
def test_fortran_character_encoding_compiles_without_injected_statements(tmp_path, flags):
    # Compile only a string constant; no generated model or adversarial code runs.
    value = "O'Brien\\'; stop ! " + "x'" * 90
    literal = fortran_string(value)
    source = tmp_path / "literal.f90"
    source.write_text("program literal\ncharacter(len=*), parameter :: name = " + literal +
                      "\nend program literal\n", encoding="utf-8")
    result = subprocess.run([shutil.which("gfortran"), *flags, "-fsyntax-only", str(source)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
