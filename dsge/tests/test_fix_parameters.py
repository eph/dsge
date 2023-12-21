import unittest
from sympy import symbols, Function
import dsge

from dsge.Base import Base, Parameter


class TestBase(unittest.TestCase):
    def setUp(self):
        self.parameters = [Parameter('beta'), Parameter('zeta'), Parameter('gamma'), Parameter('rho')]
        self.auxiliary_parameters = {Parameter('gamma'): 0.2, Parameter('kappa'): 0.1}
        self.base_obj = Base({
            'parameters': self.parameters,
            'auxiliary_parameters': self.auxiliary_parameters
        })

    def test_fix_parameters(self):
        self.base_obj.fix_parameters(beta='0.99*gamma', gamma=0.5)
        # assertions
        self.assertEqual(len(self.base_obj['parameters']), 2)
        self.assertIn(Parameter('zeta'), self.base_obj['parameters'])
        self.assertIn(Parameter('rho'), self.base_obj['parameters'])
        self.assertEqual(len(self.base_obj['auxiliary_parameters']), 3)
        self.assertEqual(self.base_obj['auxiliary_parameters'][Parameter('beta')], 0.99*Parameter('gamma'))
        self.assertEqual(self.base_obj['auxiliary_parameters'][Parameter('gamma')], 0.5)


if __name__ == "__main__":
    unittest.main()

