import numpy as np
import pandas as pd

from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.symbols import Variable, Parameter, Shock, EXP
from dsge.resource_utils import resource_path
from dsge.parsing_tools import construct_equation_list
try:
    from importlib.resources import files, as_file
except Exception:  # pragma: no cover
    from importlib_resources import files, as_file  # type: ignore

import sympy

class TestSI(TestCase):

    def setUp(self):
        with resource_path('examples/si/mankiw-reis.yaml') as p:
            self.model = read_yaml(str(p))

    def test_easy_parse(self):
     
        v = [Variable('x'), Variable('y')]
        context = {vi.name: vi for vi in v}
        context['SUM'] = lambda x,d:x
        context['oo'] = sympy.oo
        context['j'] = Parameter('j')
        context['lam'] = Parameter('lam')
        context['EXP'] = EXP
        eq2 = sympy.sympify('y+lam**j*SUM(lam*EXP(-j-1)(x-y), (j, 1, oo))', context)
        assert_equal(v[0], Variable('x'))
        print(v)
        print(eq2.atoms())
        print([(x, type(x)) for x in eq2.atoms()])
        print(eq2)

    def test_easy_parse2(self):
        # Adding an extra variable 'z' and a parameter 'k'
        v = [Variable('x'), Variable('y'), Variable('z')]
        context = {vi.name: vi for vi in v}
        context['SUM'] = sympy.Sum
        context['oo'] = sympy.oo
        context['j'] = Parameter('j')
        context['lam'] = Parameter('lam')
        context['k'] = Parameter('k')  # New parameter
        context['EXP'] = EXP
     
        # A slightly more complex expression involving the new variable and parameter
        eq = eval('lam**j*SUM(k*EXP(-j-1)(x + lam*y) + z, (j, 0, oo)) - y + z', context)
        
        assert_equal(v[0], Variable('x'))
        assert_equal(v[1], Variable('y'))
        assert_equal(v[2], Variable('z'))
        print(eq.atoms())
        print([vi.exp_date for vi in v])
    def test_more_complex_parse(self):
        v = [Variable('x'), Variable('y'), Variable('z'), Variable('w')]
        context = {vi.name: vi for vi in v}
        context['SUM'] = sympy.Sum
        context['oo'] = sympy.oo
        context['j'] = Parameter('j')
        context['lam'] = Parameter('lam')
        context['k'] = Parameter('k')
        context['EXP'] = EXP
     
        # An even more complex expression
        eq = eval('lam**j*SUM(k*EXP(-j-1)(x + lam*y) + z*EXP(j)(w + x), (j, 0, oo)) - y + z*w', context)
        
        assert_equal(v[0], Variable('x'))
        assert_equal(v[1], Variable('y'))
        assert_equal(v[2], Variable('z'))
        assert_equal(v[3], Variable('w'))
        print(eq)
        print([vi.exp_date for vi in v])

    def simple(self):
        context = {'x': Variable('x'), 'j': Parameter('j'), 'EXP': EXP}
        eq = sympy.sympify('EXP(j)(x)', locals=context)
        print(context, eq)

    def test_mankiw_reis_parsing(self):
        var_ordering= [Variable(v) for v in ['pp', 'y', 'ygr', 'delm']]
        par_ordering= [Parameter(p) for p in ['alp', 'lam', 'sigm', 'rho']]
        shk_ordering= [Shock(s) for s in ['e']]
        index = [Parameter('j')]

        equations = ['ygr + pp = -sigm*delm',
                     'alp*lam/(1-lam)*y + lam * SUM((1-lam)**j * EXP(-j-1) (pp + alp*ygr), (j, 0, oo)) = pp',
                     'ygr = y - y(-1)',
                     'delm = rho*delm(-1) + e'
            ]

        context = {}
        context['SUM'] = sympy.Sum#lambda x,d:x
        context['EXP'] = EXP
        context.update({vi.name: vi for vi in (var_ordering + par_ordering + shk_ordering + index)})

        context['oo'] = sympy.oo
        context['j'] = Parameter('j')
        context['lam'] = Parameter('lam')

        print("Before parsing:")
        for var in var_ordering:
            print(f"{var.name}: exp_date = {var.exp_date}")
     
        eq = construct_equation_list(equations, context)
     
        print("\nAfter parsing:")
        print(context)
        for var in var_ordering:
            print(f"{var.name}: exp_date = {var.exp_date}")
     
        assert_equal(var_ordering[0], Variable('pp'))
        print(eq)
        print(var_ordering)
        #assert_equal(eq[0].lhs, var_ordering[2] + var_ordering[0])

    #def test_parsing(self):
    #    assert_equal(len(self.model['var_ordering']), 4)
    #    assert_equal(self.model['var_ordering'][0], Variable('pp'))
        
