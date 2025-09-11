#!/usr/bin/env python3
"""
Tests for the FHP model validation.

This module tests the validation code specific to FHP models, including 
the constraints on future shocks.
"""

import unittest
import io
from unittest.mock import patch

from dsge.parse_yaml import read_yaml


class TestFHPValidation(unittest.TestCase):
    """Test validation specific to FHP models."""

    def setUp(self):
        """Set up test data."""
        # Define a minimal valid FHP model YAML
        self.valid_yaml = """
declarations:
  name: test_minimal_fhp
  type: fhp
  variables: [y, c, r]
  shocks: [e_a, e_b]
  innovations: [eps_a, eps_b]
  values: [v]
  value_updates: [vp]
  parameters: [alpha, beta]
  k: 1

model:
  static: []
  cycle:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  trend:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  value:
    function:
      - v = vp
    update:
      - vp = alpha*y + beta*r
  shocks:
    - e_a = 0.9*e_a(-1) + eps_a
    - e_b = 0.8*e_b(-1) + eps_b
  observables:
    y: y
    c: c
    r: r

calibration:
  parameters:
    alpha: 0.3
    beta: 0.99
  covariance:
    eps_a: 1.0
    eps_b: 1.0
"""
        
        # Define a YAML with a future shock in cycle/plan
        self.future_shock_cycle_yaml = """
declarations:
  name: test_future_shock_cycle
  type: fhp
  variables: [y, c, r]
  shocks: [e_a, e_b]
  innovations: [eps_a, eps_b]
  values: [v]
  value_updates: [vp]
  parameters: [alpha, beta]
  k: 1

model:
  static: []
  cycle:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1) + e_a(1)  # Future shock here!
  trend:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  value:
    function:
      - v = vp
    update:
      - vp = alpha*y + beta*r
  shocks:
    - e_a = 0.9*e_a(-1) + eps_a
    - e_b = 0.8*e_b(-1) + eps_b
  observables:
    y: y
    c: c
    r: r

calibration:
  parameters:
    alpha: 0.3
    beta: 0.99
  covariance:
    eps_a: 1.0
    eps_b: 1.0
"""

        # Define a YAML with a future shock in trend/terminal
        self.future_shock_trend_yaml = """
declarations:
  name: test_future_shock_trend
  type: fhp
  variables: [y, c, r]
  shocks: [e_a, e_b]
  innovations: [eps_a, eps_b]
  values: [v]
  value_updates: [vp]
  parameters: [alpha, beta]
  k: 1

model:
  static: []
  cycle:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  trend:
    terminal:
      - y = c + r
      - c = alpha*r + e_b(+1)  # Future shock here!
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  value:
    function:
      - v = vp
    update:
      - vp = alpha*y + beta*r
  shocks:
    - e_a = 0.9*e_a(-1) + eps_a
    - e_b = 0.8*e_b(-1) + eps_b
  observables:
    y: y
    c: c
    r: r

calibration:
  parameters:
    alpha: 0.3
    beta: 0.99
  covariance:
    eps_a: 1.0
    eps_b: 1.0
"""

        # Define a YAML with a future shock in value/update
        self.future_shock_value_yaml = """
declarations:
  name: test_future_shock_value
  type: fhp
  variables: [y, c, r]
  shocks: [e_a, e_b]
  innovations: [eps_a, eps_b]
  values: [v]
  value_updates: [vp]
  parameters: [alpha, beta]
  k: 1

model:
  static: []
  cycle:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  trend:
    terminal:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
    plan:
      - y = c + r
      - c = alpha*r
      - r = beta*y(-1)
  value:
    function:
      - v = vp
    update:
      - vp = alpha*y + beta*r + e_a(+1)  # Future shock here!
  shocks:
    - e_a = 0.9*e_a(-1) + eps_a
    - e_b = 0.8*e_b(-1) + eps_b
  observables:
    y: y
    c: c
    r: r

calibration:
  parameters:
    alpha: 0.3
    beta: 0.99
  covariance:
    eps_a: 1.0
    eps_b: 1.0
"""

    def test_valid_fhp_model(self):
        """Test that a valid FHP model passes validation."""
        # Suppress validation messages for this test
        with patch('builtins.print'):
            # This should not raise any exception
            model = read_yaml(io.StringIO(self.valid_yaml))
            self.assertIsNotNone(model)

    def test_future_shock_in_cycle(self):
        """Test that future shocks in cycle equations are caught."""
        with patch('builtins.print'):
            with self.assertRaises(ValueError) as context:
                read_yaml(io.StringIO(self.future_shock_cycle_yaml))
            
            error_msg = str(context.exception)
            self.assertIn("Future shocks are not allowed", error_msg)
            self.assertIn("e_a(1)", error_msg)
            self.assertIn("cycle/plan", error_msg)

    def test_future_shock_in_trend(self):
        """Test that future shocks in trend equations are caught."""
        with patch('builtins.print'):
            with self.assertRaises(ValueError) as context:
                read_yaml(io.StringIO(self.future_shock_trend_yaml))
            
            error_msg = str(context.exception)
            self.assertIn("Future shocks are not allowed", error_msg)
            self.assertIn("e_b(1)", error_msg)  # or e_b(+1)
            self.assertIn("trend/terminal", error_msg)

    def test_future_shock_in_value(self):
        """Test that future shocks in value equations are caught."""
        with patch('builtins.print'):
            with self.assertRaises(ValueError) as context:
                read_yaml(io.StringIO(self.future_shock_value_yaml))
            
            error_msg = str(context.exception)
            self.assertIn("Future shocks are not allowed", error_msg)
            self.assertIn("e_a(1)", error_msg)  # or e_a(+1)
            self.assertIn("value/update", error_msg)


if __name__ == '__main__':
    unittest.main()