import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge.DSGE import DSGE
from dsge.symbols import Variable, Equation

from numpy.testing import assert_equal

class TestEquations(TestCase):

    def test_equation(self):

        y = Variable('y')
        eq = Equation(y, 0.6*y(-1))
        print(y.date)
        var_list = [y]
        subs_dict = {}

        subs_dict.update({v:0 for v in var_list})
        subs_dict.update({v(-1):0 for v in var_list})
        subs_dict.update({v(1):0 for v in var_list})

        diff = eq.set_eq_zero.diff(y(-1)).subs(subs_dict)
        self.assertAlmostEqual(diff, -0.6)

        diff = eq.set_eq_zero.diff(y(1)).subs(subs_dict)
        self.assertAlmostEqual(diff, 0)

        diff = eq.set_eq_zero.diff(y).subs(subs_dict)
        self.assertAlmostEqual(diff, 1.0)
