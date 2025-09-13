import pytest
import sympy as sp

from dsge.parsing_tools import build_symbolic_context, parse_expression
from dsge.symbols import Variable, Parameter, Shock, EXP


def test_parse_exp_and_sum():
    x = Variable('x')
    y = Variable('y')
    j = Parameter('j')
    lam = Parameter('lam')
    ctx = build_symbolic_context([x, y, j, lam])
    expr = parse_expression('lam**j + SUM(x, (j, 0, oo))', ctx)
    assert isinstance(expr, sp.Expr)


def test_parse_exp_operator():
    x = Variable('x')
    j = Parameter('j')
    ctx = build_symbolic_context([x, j])
    expr = parse_expression('EXP(j)(x)', ctx)
    # EXP(j)(x) reduces to a Variable with exp_date=j
    assert isinstance(expr, Variable)
    assert getattr(expr, 'exp_date', None) == j


def test_unknown_symbol_raises():
    x = Variable('x')
    ctx = build_symbolic_context([x])
    with pytest.raises(ValueError):
        parse_expression('a + x', ctx)

