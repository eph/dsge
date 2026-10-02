"""Construct symbolic model expressions without evaluating Python source.

Only the mathematical DSL below is accepted.  Contexts provide symbols and
symbolic functions, never Python globals or arbitrary callable extensions.
"""
from __future__ import annotations

import ast
import keyword
import numbers
import operator
from typing import Any, Mapping

import sympy as sp
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.logic.boolalg import Boolean

from .symbols import EXP, TSymbol, Variable


def remove_sum(expression, limits):
    """Internal SI parsing mode which retains the summand only."""
    return expression


_FUNCTIONS = {
    name: getattr(sp, name) for name in (
        "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "sqrt", "Abs", "sign", "erf", "erfc",
        "Max", "Min", "Piecewise", "Heaviside", "floor", "ceiling",
    )
}
_FUNCTIONS.update({"max": sp.Max, "min": sp.Min, "abs": sp.Abs})
_CONSTANTS = {"oo": sp.oo, "inf": sp.oo, "pi": sp.pi, "E": sp.E, "I": sp.I}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow,
           ast.BitAnd: sp.And, ast.BitOr: sp.Or}
_COMPARE = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
            ast.Eq: sp.Eq, ast.NotEq: sp.Ne}


def _identifier(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name) and "__" not in name


def validate_identifier(name: str):
    """Reject source-like names even when a declaration is unused in equations."""
    if not _identifier(name):
        raise ValueError(f"Invalid mathematical identifier: {name!r}")


def declared_functions(names):
    """Create inert symbolic placeholders for explicitly declared extensions."""
    if not isinstance(names, (list, tuple)):
        raise ValueError("External function names must be a list of identifiers")
    for name in names:
        validate_identifier(name)
    return {name: sp.Function(name) for name in names}


def validate_symbolic(value):
    """Check names before symbolic input is passed to a code generator."""
    if isinstance(value, (sp.Basic, sp.MatrixBase)):
        for symbol in value.atoms(sp.Symbol):
            validate_identifier(symbol.name)
        for function in value.atoms(AppliedUndef):
            validate_identifier(function.func.__name__)
    elif isinstance(value, UndefinedFunction):
        validate_identifier(value.__name__)


def _number(value):
    if isinstance(value, bool):
        return sp.true if value else sp.false
    if isinstance(value, numbers.Integral):
        return sp.Integer(int(value))
    if isinstance(value, numbers.Real):
        return sp.Float(float(value))
    raise ValueError("Only numeric literals are permitted in expressions")


class _Builder:
    def __init__(self, context: Mapping[str, Any]):
        self.context = {**_CONSTANTS, **_FUNCTIONS, "SUM": sp.Sum, "EXP": EXP, **context}

    def lookup(self, name):
        if not _identifier(name):
            raise ValueError(f"Invalid expression identifier: {name!r}")
        if name not in self.context:
            raise ValueError(f"Unknown symbol in expression: {name}")
        return self.context[name]

    def function(self, node):
        if not isinstance(node, ast.Name):
            raise ValueError("Only named mathematical functions may be called")
        value = self.lookup(node.id)
        if isinstance(value, TSymbol):
            return value
        if any(value is fn for fn in _FUNCTIONS.values()):
            return value
        if value is sp.Sum or value is remove_sum:
            return value
        if isinstance(value, UndefinedFunction) and _identifier(value.__name__):
            return value
        raise ValueError(f"Unsupported expression function: {node.id}")

    def validate(self, node):
        """Validate the whole tree before invoking any symbolic constructor."""
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError("Only numeric literals are permitted in expressions")
        elif isinstance(node, ast.Name):
            value = self.lookup(node.id)
            if not isinstance(value, (sp.Expr, Boolean, numbers.Real)):
                raise ValueError(f"Expected a numeric value or symbol: {node.id}")
            if isinstance(value, sp.Basic):
                for symbol in value.free_symbols:
                    if not _identifier(symbol.name):
                        raise ValueError(f"Invalid symbolic identifier: {symbol.name!r}")
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self.validate(node.left)
            self.validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
            self.validate(node.operand)
        elif isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            self.validate(node.left)
            for operand in node.comparators:
                self.validate(operand)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for operand in node.values:
                self.validate(operand)
        elif isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("Keyword arguments are not permitted in expressions")
            # EXP(j)(x) is the only permitted nested call.
            if isinstance(node.func, ast.Call):
                inner = node.func
                if not (isinstance(inner.func, ast.Name) and inner.func.id == "EXP"
                        and self.lookup("EXP") is EXP and len(inner.args) == 1
                        and not inner.keywords and len(node.args) == 1):
                    raise ValueError("Only EXP(index)(expression) permits a nested call")
                self.validate(inner.args[0])
            else:
                function = self.function(node.func)
                if isinstance(function, TSymbol) and len(node.args) != 1:
                    raise ValueError("Lead/lag syntax requires one integer")
            for arg in node.args:
                if isinstance(arg, ast.Tuple):
                    # Tuple arguments occur only in SUM limits and Piecewise pairs.
                    if not (isinstance(node.func, ast.Name) and self.function(node.func) in (sp.Sum, remove_sum, sp.Piecewise)):
                        raise ValueError("Tuples are only permitted as SUM limits or Piecewise branches")
                    for element in arg.elts:
                        self.validate(element)
                else:
                    self.validate(arg)
        else:
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    def build(self, node):
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.Name):
            value = self.lookup(node.id)
            return _number(value) if isinstance(value, numbers.Real) else value
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self.build(node.left), self.build(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = self.build(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            return sp.Not(operand)
        if isinstance(node, ast.Compare):
            operands = [self.build(node.left)] + [self.build(n) for n in node.comparators]
            return sp.And(*[_COMPARE[type(op)](left, right)
                            for op, left, right in zip(node.ops, operands, operands[1:])])
        if isinstance(node, ast.BoolOp):
            return (sp.And if isinstance(node.op, ast.And) else sp.Or)(*[self.build(n) for n in node.values])
        if isinstance(node, ast.Tuple):
            return tuple(self.build(n) for n in node.elts)
        if isinstance(node, ast.Call):
            args = [self.build(arg) for arg in node.args]
            if isinstance(node.func, ast.Call):
                return EXP(self.build(node.func.args[0]))(args[0])
            function = self.function(node.func)
            if isinstance(function, TSymbol):
                if not isinstance(args[0], sp.Integer):
                    raise ValueError("Lead/lag syntax requires an integer")
            return function(*args)
        raise ValueError("Unsupported expression syntax")  # validated above


def _tree(source: str):
    try:
        # ^ is the documented alternative spelling of exponentiation.
        return ast.parse(source.strip().replace("^", "**"), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid mathematical expression syntax") from exc


def parse_expression(value, context: Mapping[str, Any]) -> sp.Expr:
    """Parse scalar arithmetic, relations, lead/lags, and explicit DSL functions.

    Python attributes, indexing, strings, comprehensions, lambdas, arbitrary
    calls, and automatic creation of unknown symbols are never accepted.
    """
    if isinstance(value, str):
        builder = _Builder(context)
        tree = _tree(value)
        builder.validate(tree)
        try:
            result = builder.build(tree)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid mathematical expression: {value!r}") from exc
    elif isinstance(value, sp.Basic):
        result = value
    else:
        result = _number(value)
    validate_symbolic(result)
    if not isinstance(result, (sp.Expr, Boolean)):
        raise ValueError("Expected a scalar mathematical expression")
    return result


def parse_matrix(value, context: Mapping[str, Any]) -> sp.Matrix:
    """Parse a rectangular nested list or Matrix([[...]]) of scalar expressions."""
    if isinstance(value, str):
        root = _tree(value)
        if isinstance(root, ast.Call) and isinstance(root.func, ast.Name) and root.func.id == "Matrix":
            if root.keywords or len(root.args) != 1:
                raise ValueError("Matrix syntax requires one nested list")
            root = root.args[0]
        if not isinstance(root, (ast.List, ast.Tuple)):
            raise ValueError("Expected a matrix as a nested list")
        builder = _Builder(context)
        rows = []
        for row in root.elts:
            if not isinstance(row, (ast.List, ast.Tuple)):
                raise ValueError("Expected a matrix as a nested list")
            for entry in row.elts:
                builder.validate(entry)
            rows.append(row.elts)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Matrix must be non-empty and rectangular")
        return sp.Matrix([[builder.build(entry) for entry in row] for row in rows])
    if isinstance(value, sp.MatrixBase):
        validate_symbolic(value)
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Expected a matrix as a nested list")
    if any(not isinstance(row, (list, tuple)) for row in value):
        raise ValueError("Expected a matrix as a nested list")
    if not value[0] or any(len(row) != len(value[0]) for row in value):
        raise ValueError("Matrix must be non-empty and rectangular")
    return sp.Matrix([[parse_expression(entry, context) for entry in row] for row in value])


def parse_variable_label(value: str, context: Mapping[str, Any]) -> Variable:
    """Accept just a declared variable name, optionally with an integer shift."""
    tree = _tree(value)
    if not isinstance(tree, ast.Name) and not (
        isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name)
        and len(tree.args) == 1 and not tree.keywords
    ):
        raise ValueError("Column labels must name a model variable, optionally with a lead/lag")
    if isinstance(tree, ast.Call):
        shift = tree.args[0]
        if isinstance(shift, ast.UnaryOp) and isinstance(shift.op, (ast.UAdd, ast.USub)):
            shift = shift.operand
        if not (isinstance(shift, ast.Constant) and type(shift.value) is int):
            raise ValueError("Column lead/lag must be an integer literal")
    result = parse_expression(value, context)
    if not isinstance(result, Variable):
        raise ValueError("Column labels must name a model variable")
    return result


def prepare_expression(value, context: Mapping[str, Any]):
    """Ensure Base.lambdify never receives unparsed strings, even in containers."""
    if isinstance(value, str):
        tree = _tree(value)
        if isinstance(tree, (ast.List, ast.Tuple)) or (
            isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == "Matrix"
        ):
            return parse_matrix(value, context)
        return parse_expression(value, context)
    if isinstance(value, sp.MatrixBase):
        validate_symbolic(value)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(prepare_expression(entry, context) for entry in value)
    if hasattr(value, "tolist"):
        return prepare_expression(value.tolist(), context)
    return parse_expression(value, context)
