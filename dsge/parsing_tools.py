import sympy as sp
from typing import List, Dict, Union, Any, Iterable, Optional
from .symbols import Equation, Parameter, Shock, Variable
from .symbols import EXP
import itertools

def build_symbolic_context(
    symbols: Optional[Iterable[Union[Parameter, Variable, Shock]]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a safe, consistent context for parsing model expressions.

    Includes model symbols, common SymPy math functions, and DSL helpers.
    """
    ctx: Dict[str, Any] = {}
    if symbols:
        ctx.update({s.name: s for s in symbols})

    # Common math and constants
    ctx.update({
        'exp': sp.exp,
        'log': sp.log,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'asin': sp.asin,
        'acos': sp.acos,
        'atan': sp.atan,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'sqrt': sp.sqrt,
        'Abs': sp.Abs,
        'sign': sp.sign,
        'oo': sp.oo,
    })

    # DSL helpers
    ctx['EXP'] = EXP
    ctx['SUM'] = sp.Sum

    # User extras
    if extras:
        ctx.update(extras)

    return ctx


def parse_expression(expr: str, context: Dict[str, Any]) -> sp.Expr:
    """Parse a string expression into a SymPy expression using a safe context.

    Uses SymPy's sympify with a whitelisted local context. Validates that
    all model symbols used in the expression are present in the context.
    """
    expr_obj = sp.sympify(expr, locals=context)

    symbols_in_expr = expr_obj.atoms(Parameter) | expr_obj.atoms(Variable) | expr_obj.atoms(Shock)
    for symbol in symbols_in_expr:
        if symbol.name not in context:
            raise ValueError(
                f"In expression {expr}, symbol {symbol} is not in the provided context"
            )

    # Also ensure no bare SymPy symbols slipped in that aren't known in context
    unknown_free = [s for s in expr_obj.free_symbols if s.name not in context]
    if unknown_free:
        raise ValueError(f"Unknown symbols in expression {expr}: {unknown_free}")
    return expr_obj

def from_dict_to_mat(dictionary: Dict[str, str], element_array: List[str], context: Dict[str, Any], is_symmetric: bool =True) -> sp.Matrix:
    """
    This function takes a dictionary, an array of elements, and a context.
    It then transforms the dictionary into a SymPy matrix based on the element_array and evaluates
    dictionary values using the context provided.

    Parameters
    ----------
    dictionary: Dict[str, str]
        A dictionary where the keys are expected to be strings that represent either single shocks (e.g., 'var')
        or pairs of shocks (e.g., 'var1,var2'), and the values are string expressions that can be evaluated
        in the provided context.
    element_array: List[str]
        List of elements that correspond to the keys in the dictionary.
    context: Dict[str, Any]
        A dictionary that maps variable names to values.
        This context will be used to evaluate the string expressions from the input dictionary.

    Returns
    -------
    sp.Matrix
        A SymPy matrix that has been populated with the evaluated expressions from the dictionary.
        The matrix uses element_array for positioning the evaluated values. Values with single shock
        keys go in the diagonal, while pairs of shock keys go in the i, j and j, i positions respectively.

    Raises
    ------
    ValueError
        If a key in the dictionary has other than one or two variables (shocks).
    """
    ns = len(element_array)
    matrix = sp.zeros(ns, ns)

    type_of_array = type(element_array[0])
    for key, value in dictionary.items():
        shocks = key.split(',')

        if len(shocks) == 1:
            i = element_array.index(type_of_array(shocks[0].strip()))
            matrix[i, i] = parse_expression(str(value), context)
        elif len(shocks) == 2:
            i = element_array.index(type_of_array(shocks[0].strip()))
            j = element_array.index(type_of_array(shocks[1].strip()))
            matrix[i, j] = parse_expression(str(value), context)

            if is_symmetric:
                matrix[j, i] = matrix[i, j]
        else:
            raise ValueError('The key {} is not valid'.format(key))
    return matrix

def construct_equation_list(raw_equations, context):
    equations = []

    for eq in raw_equations:
        if "=" in eq:
            lhs, rhs = str.split(eq, "=")
        else:
            lhs, rhs = eq, "0"

        try:
            lhs = parse_expression(lhs, context)
            rhs = parse_expression(rhs, context)
        except TypeError as e:
            print("While parsing %s, got this error: %s" % (eq, repr(e)))

        equations.append(Equation(lhs,rhs))

    return equations

def find_max_lead_lag(equations, shocks_or_variables):
    it = itertools.chain.from_iterable

    max_lead = dict.fromkeys(shocks_or_variables)
    max_lag = dict.fromkeys(shocks_or_variables)

    type_of_array = type(shocks_or_variables[0])
    all_shocks = [list(eq.atoms(type_of_array)) for eq in equations]

    for s in shocks_or_variables:
        max_lead[s] = max([i.date for i in it(all_shocks) if i.name == s.name])
        max_lag[s] = min([i.date for i in it(all_shocks) if i.name == s.name])

    return max_lead, max_lag

def parse_calibration(calibration: Dict,
                      parameters: List[Parameter],
                      auxiliary_parameters: List[Parameter],
                      shocks: List[sp.Symbol]) -> Dict[str, Dict[str, float]]:

    calibration_dict = dict()

    # starting values
    calibration_dict['parameters'] = dict()
    for p in parameters:
        calibration_dict['parameters'][p] = float(calibration['parameters'][str(p)])

    # auxiliary parameters
    #calibration_dict['auxiliary_parameters'] = {p
    
