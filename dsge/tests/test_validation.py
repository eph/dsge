#!/usr/bin/env python3
"""
Tests for the validation module.

This module contains unit tests for the validation functions in the dsge.validation module.
"""

import unittest
from unittest.mock import MagicMock, patch
import sympy

from dsge.symbols import Variable, Shock, Parameter, Equation
from dsge.validation import (
    find_symbols_in_equation,
    find_future_symbols,
    check_for_future_shocks,
    find_max_lead_lag,
    validate_model_consistency
)


class TestValidation(unittest.TestCase):
    """Test suite for the validation module."""

    def setUp(self):
        """Set up test data before each test."""
        # Create some test variables and symbols
        self.x = Variable('x')
        self.y = Variable('y')
        self.z = Variable('z')
        self.e_a = Variable('e_a')  # shock variable
        self.e_b = Variable('e_b')  # shock variable
        
        # Create variables with leads and lags
        self.x_lag = self.x(-1)    # x(t-1)
        self.x_fut = self.x(1)     # x(t+1)
        self.y_lag = self.y(-1)    # y(t-1)
        self.y_fut = self.y(1)     # y(t+1)
        self.e_a_lag = self.e_a(-1)  # e_a(t-1)
        self.e_a_fut = self.e_a(1)   # e_a(t+1)
        
        # Create some equations
        self.eq1 = Equation(self.x, self.y + self.z)  # x = y + z
        self.eq2 = Equation(self.x, self.y_lag + self.z)  # x = y(-1) + z
        self.eq3 = Equation(self.x, self.y_fut + self.z)  # x = y(+1) + z
        self.eq4 = Equation(self.x, self.e_a + self.y)  # x = e_a + y
        self.eq5 = Equation(self.x, self.e_a_fut + self.y)  # x = e_a(+1) + y
        
        # List of variables and shocks
        self.variables = [self.x, self.y, self.z]
        self.shocks = [self.e_a, self.e_b]
        
        # List of equations
        self.equations = [self.eq1, self.eq2, self.eq3, self.eq4, self.eq5]

    def test_find_symbols_in_equation(self):
        """Test finding symbols in equations."""
        # Find all variables in eq1
        vars_in_eq1 = find_symbols_in_equation(self.eq1, Variable)
        self.assertEqual(len(vars_in_eq1), 3)
        self.assertIn(self.x, vars_in_eq1)
        self.assertIn(self.y, vars_in_eq1)
        self.assertIn(self.z, vars_in_eq1)
        
        # Find specific variables in eq4
        vars_in_eq4 = find_symbols_in_equation(self.eq4, Variable, ['e_a'])
        self.assertEqual(len(vars_in_eq4), 1)
        self.assertEqual(vars_in_eq4[0].name, 'e_a')
        
        # Find variables with restricted names
        vars_in_eq3 = find_symbols_in_equation(self.eq3, Variable, ['x', 'y'])
        self.assertEqual(len(vars_in_eq3), 2)
        # Note: this contains y(+1), not y

    def test_find_future_symbols(self):
        """Test finding future symbols in equations."""
        # Find future "y" variables
        result = find_future_symbols([self.eq3], [self.y])
        self.assertEqual(len(result), 1)
        self.assertIn(self.eq3, result)
        symbols_dates = result[self.eq3]
        self.assertEqual(len(symbols_dates), 1)
        self.assertEqual(symbols_dates[0][0].name, 'y')
        self.assertEqual(symbols_dates[0][1], 1)  # date = 1
        
        # Find future shocks
        result = find_future_symbols([self.eq5], [self.e_a])
        self.assertEqual(len(result), 1)
        self.assertIn(self.eq5, result)
        symbols_dates = result[self.eq5]
        self.assertEqual(len(symbols_dates), 1)
        self.assertEqual(symbols_dates[0][0].name, 'e_a')
        self.assertEqual(symbols_dates[0][1], 1)  # date = 1
        
        # Check an equation with no future symbols
        result = find_future_symbols([self.eq1, self.eq2], [self.x, self.y, self.z])
        self.assertEqual(len(result), 0)

    def test_check_for_future_shocks(self):
        """Test checking for future shocks."""
        # Mock function for getting original equation
        get_original_eq_fn = MagicMock(return_value="original equation")
        
        # Test with equation that has no future shocks
        try:
            check_for_future_shocks([self.eq1, self.eq2, self.eq4], [self.e_a, self.e_b], 
                                   "test_section", get_original_eq_fn)
            # Should not raise an exception
        except ValueError:
            self.fail("check_for_future_shocks raised ValueError unexpectedly!")
        
        # Test with equation that has future shocks - should raise ValueError
        with self.assertRaises(ValueError) as context:
            check_for_future_shocks([self.eq5], [self.e_a, self.e_b], 
                                  "test_section", get_original_eq_fn)
        
        error_msg = str(context.exception)
        self.assertIn("Future shocks are not allowed", error_msg)
        self.assertIn("e_a(1)", error_msg)

    def test_find_max_lead_lag(self):
        """Test finding max lead and lag for variables."""
        # Create equations with various leads and lags
        eq_complex = Equation(
            self.x(2),  # x(t+2)
            self.y(-3) + self.z(1)  # y(t-3) + z(t+1)
        )
        
        max_lead, max_lag = find_max_lead_lag(
            [eq_complex], [self.x, self.y, self.z]
        )
        
        # Check lead values
        self.assertEqual(max_lead[self.x], 2)
        self.assertEqual(max_lead[self.y], 0)
        self.assertEqual(max_lead[self.z], 1)
        
        # Check lag values (note our function returns negative values for lags)
        self.assertEqual(max_lag[self.x], 0)
        self.assertEqual(max_lag[self.y], -3)
        self.assertEqual(max_lag[self.z], 0)

    def test_validate_model_consistency(self):
        """Test model consistency validation."""
        # Create a valid model
        valid_model = {
            'variables': [self.x, self.y, self.z],
            'equations': {
                'main': [self.eq1, self.eq2]
            }
        }
        
        warnings = validate_model_consistency(valid_model)
        self.assertEqual(len(warnings), 0)
        
        # Create a model with undeclared variables
        w = Variable('w')
        eq_invalid = Equation(self.x, w + self.y)
        
        invalid_model = {
            'variables': [self.x, self.y, self.z],
            'equations': {
                'main': [eq_invalid]
            }
        }
        
        warnings = validate_model_consistency(invalid_model)
        self.assertEqual(len(warnings), 1)
        self.assertIn("w", warnings[0])


if __name__ == '__main__':
    unittest.main()