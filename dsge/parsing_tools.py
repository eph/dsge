import sympy as sp
from typing import List, Dict, Tuple, Union, Callable, Optional, Any
from .symbols import Equation, Parameter, Shock, Variable
import itertools

def parse_expression(expr: str, context: Dict[str, Union[Parameter, Variable, Shock]]) -> sp.Expr:

    expr = sp.sympify(expr, locals=context)

    symbols_in_expr = expr.atoms(Parameter) | expr.atoms(Variable) | expr.atoms(Shock)
    for symbol in symbols_in_expr:
        if symbol.name not in context:
            raise ValueError(f"In expression {expr}, symbol {symbol} is not in the context {context}")
    return expr

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
    
