#!/usr/bin/env python3
"""
Tests for the DSGE model validation.

This module tests the validation code specific to standard DSGE models.
"""

import unittest
import io
import yaml
import sympy
from unittest.mock import patch

from dsge.parse_yaml import read_yaml
from dsge.symbols import Variable, Shock, Parameter, Equation
from dsge.validation import validate_dsge_leads_lags


class TestDSGEValidation(unittest.TestCase):
    """Test validation specific to standard DSGE models."""

    def setUp(self):
        """Set up test data."""
        # Create some test variables
        self.x = Variable('x')
        self.y = Variable('y')
        self.z = Variable('z')
        
        # Variables with leads and lags
        self.x_lag1 = self.x(-1)
        self.x_lag2 = self.x(-2)
        self.x_lead1 = self.x(1)
        self.x_lead2 = self.x(2)
        
        self.y_lag1 = self.y(-1)
        self.y_lag2 = self.y(-2)
        self.y_lead1 = self.y(1)
        self.y_lead2 = self.y(2)
        
        # Create some equations
        self.eq1 = Equation(self.x, self.y + self.z)  # x = y + z (valid)
        self.eq2 = Equation(self.x, self.y_lag1 + self.z)  # x = y(-1) + z (valid)
        self.eq3 = Equation(self.x, self.y_lead1 + self.z)  # x = y(+1) + z (valid)
        self.eq4 = Equation(self.x, self.y_lag2 + self.z)  # x = y(-2) + z (lag exceeds 1)
        self.eq5 = Equation(self.x, self.y_lead2 + self.z)  # x = y(+2) + z (lead exceeds 1)
        
        # List of variables
        self.variables = [self.x, self.y, self.z]
        
        # YAML for a valid DSGE model
        self.valid_yaml = """
declarations:
  name: test_dsge
  variables: [x, y, z]
  shocks: [e_x, e_y]
  parameters: [alpha, beta]

equations:
  model:
    - x = alpha*x(-1) + e_x
    - y = beta*y(-1) + x + e_y
    - z = alpha*x + beta*y
  observables:
    x: x
    y: y
    z: z

calibration:
  parameters:
    alpha: 0.5
    beta: 0.9
  covariance:
    e_x: 1.0
    e_y: 1.0
"""
        
        # YAML for a DSGE model with excessive lags
        self.excessive_lag_yaml = """
declarations:
  name: test_dsge_lag
  variables: [x, y, z]
  shocks: [e_x, e_y]
  parameters: [alpha, beta]

equations:
  model:
    - x = alpha*x(-1) + e_x
    - y = beta*y(-2) + x + e_y  # y(-2) exceeds max lag
    - z = alpha*x + beta*y
  observables:
    x: x
    y: y
    z: z

calibration:
  parameters:
    alpha: 0.5
    beta: 0.9
  covariance:
    e_x: 1.0
    e_y: 1.0
"""
        
        # YAML for a DSGE model with excessive leads
        self.excessive_lead_yaml = """
declarations:
  name: test_dsge_lead
  variables: [x, y, z]
  shocks: [e_x, e_y]
  parameters: [alpha, beta]

equations:
  model:
    - x = alpha*x(2) + e_x  # x(2) exceeds max lead
    - y = beta*y(-1) + x + e_y
    - z = alpha*x + beta*y
  observables:
    x: x
    y: y
    z: z

calibration:
  parameters:
    alpha: 0.5
    beta: 0.9
  covariance:
    e_x: 1.0
    e_y: 1.0
"""

    def test_dsge_validation_function(self):
        """Test the validate_dsge_leads_lags function."""
        # Valid case: max_lead=1, max_lag=1
        errors = validate_dsge_leads_lags(
            [self.eq1, self.eq2, self.eq3],
            self.variables,
            max_lead=1,
            max_lag=1
        )
        self.assertEqual(len(errors), 0)
        
        # Invalid case: lag violation
        errors = validate_dsge_leads_lags(
            [self.eq1, self.eq4],
            self.variables,
            max_lead=1,
            max_lag=1
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds maximum lag", errors[0])
        self.assertIn("y(-2)", errors[0])
        
        # Invalid case: lead violation
        errors = validate_dsge_leads_lags(
            [self.eq5],
            self.variables,
            max_lead=1,
            max_lag=1
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds maximum lead", errors[0])
        self.assertIn("y(2)", errors[0])
        
        # Higher limits should pass
        errors = validate_dsge_leads_lags(
            [self.eq1, self.eq2, self.eq3, self.eq4, self.eq5],
            self.variables,
            max_lead=2,
            max_lag=2
        )
        self.assertEqual(len(errors), 0)

    # Focus on testing the validation functions directly rather than through the model parser
    # These tests will need to be updated when we integrate the validation into the DSGE class
    def test_parse_yaml_without_validation(self):
        """Test that we can parse a YAML file without triggering validation."""
        # We'll just verify we can parse the YAML, not that the model is valid
        yaml_dict = yaml.safe_load(self.valid_yaml)
        self.assertIsNotNone(yaml_dict)
        self.assertEqual(yaml_dict['declarations']['name'], 'test_dsge')
        
        yaml_dict = yaml.safe_load(self.excessive_lag_yaml)
        self.assertIsNotNone(yaml_dict)
        self.assertEqual(yaml_dict['declarations']['name'], 'test_dsge_lag')
        
        yaml_dict = yaml.safe_load(self.excessive_lead_yaml)
        self.assertIsNotNone(yaml_dict)
        self.assertEqual(yaml_dict['declarations']['name'], 'test_dsge_lead')


if __name__ == '__main__':
    unittest.main()