#!/usr/bin/env python3
import sympy as sp
from typing import Dict, List, Any
from .symbols import Equation
import itertools

import yaml
import re

def read_yaml(yaml_file, sub_list=[('^', '**'), (';','')]):
    """
    This function reads a
    """
    with open(yaml_file) as f:
        txt = f.read()

    for old, new in sub_list:
        txt = txt.replace(old, new)

    txt = re.sub(r"@ ?\n", " ", txt)

    return yaml.safe_load(txt)


def from_dict_to_mat(dictionary: Dict[str, str], element_array: List[str], context: Dict[str, Any]) -> sp.Matrix:
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
            matrix[i, i] = eval(str(value), context)
        elif len(shocks) == 2:
            i = element_array.index(type_of_array(shocks[0].strip()))
            j = element_array.index(type_of_array(shocks[1].strip()))
            matrix[i, j] = eval(str(value), context)
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
            lhs = eval(lhs, context)
            rhs = eval(rhs, context)
        except TypeError as e:
            print("While parsing %s, got this error: %s" % (eq, repr(e)))
            return

        equations.append(Equation(sp.sympify(lhs), sp.sympify(rhs)))
    return equations

def find_max_lead_lag(equations, shocks_or_variables):
    it = itertools.chain.from_iterable

    max_lead_exo = dict.fromkeys(shocks_or_variables)
    max_lag_exo = dict.fromkeys(shocks_or_variables)

    type_of_array = type(shocks_or_variables[0])
    all_shocks = [list(eq.atoms(type_of_array)) for eq in equations]

    for s in shocks_or_variables:
        max_lead_exo[s] = max([i.date for i in it(all_shocks) if i.name == s.name])
        max_lag_exo[s] = min([i.date for i in it(all_shocks) if i.name == s.name])

    return max_lead_exo, max_lag_exo
