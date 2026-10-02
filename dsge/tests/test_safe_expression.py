"""Security boundaries for model expressions; payloads have no external effects."""
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sympy as sp
import yaml

from dsge import read_yaml
from dsge.Base import Base
from dsge.DSGE import DSGE
from dsge.FHPRepAgent import FHPRepAgent
from dsge.MarkovSwitching import MarkovSwitchingModel
from dsge.SIDSGE import read_si
from dsge.irfoc import IRFOC, parse_policy_rules
from dsge.obc import OccBinModel
from dsge.oc import parse_loss
from dsge.parsing_tools import build_symbolic_context, parse_expression
from dsge.read_mod import read_mod
from dsge.safe_expression import parse_matrix, parse_variable_label, remove_sum
from dsge.switching_ssm import _parse_expr
from dsge.symbols import Parameter, Variable


BAD_SYNTAX = [
    "__import__('os').getcwd()", "x.__class__", "[x][0]",
    "(lambda: 0)()", "[x for x in (1,)]", "unknown(x)",
    "Symbol('x')", "getattr(x, 'name')", "x if True else 0",
    "exp(x, evaluate=False)", "(x := 1)", "{'x': 1}",
]


def simple_model_data():
    return {
        "declarations": {"name": "safe", "variables": ["x"],
                         "parameters": ["rho"], "shocks": ["e"]},
        "equations": ["x = rho*x(-1) + e"],
        "calibration": {"parameters": {"rho": 0.8}},
    }


@pytest.mark.parametrize("source", BAD_SYNTAX)
def test_rejects_python_syntax(source):
    x = Variable("x")
    with pytest.raises(ValueError):
        parse_expression(source, build_symbolic_context([x]))


def test_context_callable_is_never_invoked_and_entire_ast_is_validated():
    touch = Mock(return_value=0)
    with pytest.raises(ValueError):
        parse_expression("touch()", {"touch": touch})
    touch.assert_not_called()
    # Even an allowed symbolic function is not constructed before the rest of
    # the input has been checked. The callback here is deliberately not allowed.
    with pytest.raises(ValueError):
        parse_expression("touch() + x.__class__", {"touch": touch, "x": Variable("x")})
    touch.assert_not_called()


def test_math_leads_expectations_sums_and_conditions():
    x, j, beta = Variable("x"), Parameter("j"), Parameter("beta")
    context = build_symbolic_context([x, j, beta])
    assert parse_expression("x(-2) + beta^2*x(+1)", context) == x(-2) + beta**2*x(1)
    assert parse_expression("exp(log(beta)) + sqrt(4)", context) == beta + 2
    assert parse_expression("EXP(-j-1)(x)", context).exp_date == -j-1
    assert isinstance(parse_expression("SUM(beta**j*x, (j, 0, oo))", context), sp.Sum)
    assert parse_expression("SUM(beta**j*x, (j, 0, oo))", {**context, "SUM": remove_sum}) == beta**j*x
    assert parse_expression("(x < 0) | (x >= 1)", context) == sp.Or(x < 0, x >= 1)
    assert parse_expression("Piecewise((x, x > 0), (0, True))", context) == sp.Piecewise((x, x > 0), (0, True))
    with pytest.raises(ValueError, match="integer"):
        parse_expression("x(0.5)", context)


def test_matrix_and_label_profiles():
    p, x = Parameter("p"), Variable("x")
    expected = sp.Matrix([[p, 1-p], [0, 1]])
    for source in ("[[p, 1-p], [0, 1]]", "Matrix([[p, 1-p], [0, 1]])", [[p, "1-p"], [0, 1]]):
        assert parse_matrix(source, {"p": p}) == expected
    assert parse_matrix(np.eye(2), {}) == sp.Matrix(np.eye(2))
    assert parse_variable_label("x(-1)", {"x": x}) == x(-1)
    for source in ("x + 1", "exp(x)", "x.__class__", "[x][0]"):
        with pytest.raises(ValueError):
            parse_variable_label(source, {"x": x})
    for source in ("[[p], [0, 1]]", "[p, 1]", "Matrix(2, 2, lambda i,j: i)"):
        with pytest.raises(ValueError):
            parse_matrix(source, {"p": p})


def test_fixed_parameters_parse_transactionally_and_lambdify_is_guarded():
    rho, beta = Parameter("rho"), Parameter("beta")
    model = Base(parameters=[rho, beta], auxiliary_parameters={})
    with pytest.raises(ValueError):
        model.fix_parameters(rho="__import__('os').getcwd()")
    assert model["parameters"] == [rho, beta]
    assert model["auxiliary_parameters"] == {}
    model.fix_parameters(rho="beta/2")
    assert model.lambdify("rho + beta")([0.8]) == pytest.approx(1.2)
    assert np.asarray(model.lambdify("[[rho, beta]]")([0.8])) == pytest.approx(np.array([[0.4, 0.8]]))
    for value in ("__import__('os').getcwd()", ["__import__('os').getcwd()"]):
        with pytest.raises(ValueError):
            model.lambdify(value)


def test_entrypoints_reject_payload_without_creating_file(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    model = DSGE.read(simple_model_data())
    x = model.variables[0]
    context = {"x": x}
    calls = [
        lambda: parse_policy_rules(f"x = {payload}", variables=[x], variable_context=context),
        lambda: parse_loss(payload, [x], [], []),
        lambda: _parse_expr(payload, ctx=context, allowed_symbols={x}, where="test"),
        lambda: OccBinModel(regimes={"normal": model}, constraint={"when": payload}, yaml=simple_model_data()),
        lambda: MarkovSwitchingModel(model, model, f"[[{payload}, 0], [0, 1]]"),
        lambda: IRFOC(model, pd.DataFrame({payload: [0.0]}), "e", compiled_model=Mock()),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()
        assert not sentinel.exists()

    data = simple_model_data()
    data["equations"] = [f"x = {payload}"]
    with pytest.raises(ValueError):
        read_yaml(StringIO(yaml.safe_dump(data)))
    assert not sentinel.exists()
    path = tmp_path / "unsafe.mod"
    path.write_text(f"var x; varexo e; parameters rho; rho=0.8; model; x={payload}; end;")
    with pytest.raises(ValueError):
        read_mod(path)
    assert not sentinel.exists()


def test_fhp_and_si_auxiliary_parameter_strings_rejected(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    root = Path(__file__).parents[1] / "examples"
    fhp = yaml.safe_load((root / "fhp" / "partial_equilibrium_endogenous.yaml").read_text())
    fhp["declarations"]["auxiliary_parameters"] = ["unsafe_aux"]
    fhp["calibration"]["auxiliary_parameters"] = {"unsafe_aux": payload}
    with pytest.raises(ValueError):
        FHPRepAgent.read(fhp)
    si = yaml.safe_load((root / "si" / "mankiw-reis.yaml").read_text())
    si["declarations"]["auxiliary_parameters"] = ["unsafe_aux"]
    si["calibration"]["auxiliary_parameters"] = {"unsafe_aux": payload}
    with pytest.raises(ValueError):
        read_si(si)
    assert not sentinel.exists()


@pytest.mark.parametrize("name", ["__import__", "x;print(1)", "x[0]", "lambda"])
def test_unused_declared_and_prebuilt_symbol_names_rejected(name):
    bad = Parameter(name)
    with pytest.raises(ValueError):
        build_symbolic_context([bad])
    with pytest.raises(ValueError):
        Base(parameters=[bad], auxiliary_parameters={}).lambdify(sp.Integer(0))
    with pytest.raises(ValueError):
        parse_expression(bad + 1, {})
    with pytest.raises(ValueError):
        parse_matrix(sp.Matrix([[bad]]), {})
    with pytest.raises(ValueError):
        parse_expression(sp.Function(name)(1), {})


def test_no_eval_capable_string_parser_in_production_modules():
    # Keep a small source-level tripwire alongside the behavioral API tests.
    import ast
    for path in Path(__file__).parents[1].glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name == "sympify":
                assert node.args and isinstance(node.args[0], ast.Constant), path
                assert type(node.args[0].value) in (int, float), path
            if isinstance(node.func, ast.Name):
                assert name not in ("eval", "exec"), path


@pytest.mark.parametrize("kind", ["fhp", "switching"])
@pytest.mark.parametrize("field", ["lambda", "policy_object", "cost"])
def test_horizon_strings_reject_code(kind, field, tmp_path):
    sentinel = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    root = Path(__file__).parents[1] / "examples"
    filename = root / ("fhp/partial_equilibrium_endogenous.yaml" if kind == "fhp"
                       else "switching/pe_pricing_switching.yaml")
    data = yaml.safe_load(filename.read_text())
    key = "stopping_rule" if kind == "fhp" else "horizon_choice"
    cfg = data["declarations"][key]["components"]["pricing"]
    cfg[field] = {"a": payload} if field == "cost" else payload
    with pytest.raises(ValueError):
        model = read_yaml(StringIO(yaml.safe_dump(data)))
        if kind == "fhp":
            model.compile_model()
    assert not sentinel.exists()


def test_irfoc_losses_and_discount_strings_reject_code(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    model = DSGE.read(simple_model_data())
    control = IRFOC(model, pd.DataFrame({"x": [0.0, 0.0]}), "e")
    for loss, discount in ((payload, 1.0), ("x**2", payload)):
        with pytest.raises(ValueError):
            control.simulate_optimal_control(loss, discount=discount)
    assert not sentinel.exists()


@pytest.mark.parametrize("source", [
    "0.12345678901234567890123456789", "1e-1000", "1e1000", "0.1",
    "2.5e-17", "1_0.2_5", "0xff", "0b101", "1_000",
])
def test_numeric_literals_preserve_sympy_precision(source):
    expected = sp.Float(source.replace("_", "")) if any(c in source for c in ".eE") else sp.Integer(int(source, 0))
    actual = parse_expression(source, {})
    assert actual == expected
    if isinstance(expected, sp.Float):
        assert actual._prec == expected._prec
        assert actual.is_finite is True
        assert actual != 0


def test_named_symbolic_numbers_are_not_rounded_to_python_float():
    values = {"third": sp.Rational(1, 3),
              "precise": sp.Float("0.12345678901234567890", 30)}
    for name, value in values.items():
        assert parse_expression(name, values) is value
    assert parse_expression("third + precise", values) == values["third"] + values["precise"]


def test_numeric_literals_preserve_arithmetic_and_matrix_semantics():
    # Tiny/large literals can cancel to an ordinary finite number before numeric
    # code generation; rounding each token through Python float destroys this.
    tiny = sp.Float("1e-1000")
    large = sp.Float("1e1000")
    assert parse_expression("1e-1000 * 1e1000", {}) == tiny * large
    expected = sp.Float("0.12345678901234567890123456789")
    assert parse_matrix("[[0.12345678901234567890123456789]]", {})[0, 0] == expected
    assert parse_expression("-2^2 + 1/3 + (4 + 5)*6", {}) == -4 + sp.Rational(1, 3) + 54


@pytest.mark.parametrize("source, expected", [
    ("[p, 1-p]", [0.25, 0.75]),
    ("(p, 1-p)", (0.25, 0.75)),
    ("[[p, 1-p]]", [[0.25, 0.75]]),
    ("[p, (1-p, [p**2])]", [0.25, (0.75, [0.0625])]),
    ("[]", []),
    ("()", ()),
])
def test_lambdify_string_containers_preserve_structure_and_types(source, expected):
    model = Base(parameters=[Parameter("p")], auxiliary_parameters={})
    actual = model.lambdify(source)([0.25])

    def assert_structure(left, right):
        assert type(left) is type(right)
        if isinstance(right, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_structure(a, b)
        else:
            assert left == right

    assert_structure(actual, expected)


def test_lambdify_nested_container_payload_is_rejected_without_effects(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    payload = f"[0, (1, [__import__('pathlib').Path({str(sentinel)!r}).touch()])]"
    model = Base(parameters=[], auxiliary_parameters={})
    with pytest.raises(ValueError):
        model.lambdify(payload)
    assert not sentinel.exists()
