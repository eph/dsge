#!/usr/bin/env python3
"""
Display a DSGE model with enhanced string formatting.

This script loads and displays a DSGE model from the example files,
showing how the model's equations and variables are represented with
the enhanced Unicode formatting.
"""

import os
import sys
import re

# Make sure to import from the local implementation
sys.path.insert(0, os.path.abspath('.'))

# First try the basic NKMP model
from dsge.parse_yaml import read_yaml
from dsge.symbols import Equation, Parameter, Variable, convert_to_greek

def format_equation(eq_str):
    """Format equation using enhanced pretty printing rules."""
    # Step 1: Detect and convert any Greek parameter names
    for greek_name, symbol in convert_to_greek.__globals__['GREEK_LETTERS'].items():
        pattern = r'\b' + re.escape(greek_name) + r'\b'
        eq_str = re.sub(pattern, symbol, eq_str, flags=re.IGNORECASE)
    
    # Step 2: Replace * with · for multiplication
    eq_str = eq_str.replace('*', '·')
    
    # Step 3: Format time indices for variables
    # Convert x(+1) to x_{t+1} and x(-1) to x_{t-1}
    eq_str = re.sub(r'(\w+)\(([+-]?\d+)\)', lambda m: f"{m.group(1)}_{{t{m.group(2)}}}", eq_str)
    
    # Step 4: Format expectations
    eq_str = eq_str.replace('E[', '𝔼_t[')
    
    # Step 5: Handle exponents
    eq_str = re.sub(r'(\w+)\^2', r'\1²', eq_str)
    eq_str = re.sub(r'(\w+)\^3', r'\1³', eq_str)
    
    return eq_str

def display_model(model_path):
    """Load and display a DSGE model."""
    print(f"Loading model from: {model_path}")
    try:
        model = read_yaml(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try to find a simpler model
        if "nkmp" in model_path:
            simpler_path = "dsge/examples/ar1/ar1.yaml"
            if os.path.exists(simpler_path):
                print(f"Falling back to simpler model: {simpler_path}")
                model = read_yaml(simpler_path)
            else:
                raise
        else:
            raise
    
    print("\n=== Model Details ===")
    print(f"Model Name: {model.name if hasattr(model, 'name') else model['declarations'].get('name', 'Unnamed model')}")
    print(f"Model Type: {model['declarations'].get('type', 'dsge')}")
    
    print("\n=== Variables ===")
    for var in model['variables']:
        print(f"  {var} ({var.name})")
    
    print("\n=== Parameters ===")
    for param in model['parameters']:
        print(f"  {param} ({param.name})")
    
    print("\n=== Equations ===")
    eq_list = model['equations']
    for i, eq in enumerate(eq_list):
        # Give each equation a name based on its index
        eq_name = f"Equation {i+1}"
        
        # Create a proper Equation object for nice formatting
        if hasattr(eq, 'lhs') and hasattr(eq, 'rhs'):
            # Already an equation object
            formatted_eq = Equation(eq.lhs, eq.rhs, name=eq_name)
        else:
            # Get the original equation string from the YAML file if possible
            try:
                if hasattr(model, 'yaml_equations') and i < len(model.yaml_equations):
                    eq_str = model.yaml_equations[i]
                    eq_str = format_equation(eq_str)
                    print(f"  {eq_name}: {eq_str}")
                    continue
                else:
                    # Just format the string representation directly
                    eq_str = str(eq)
                    eq_str = format_equation(eq_str)
                    print(f"  {eq_name}: {eq_str}")
                    continue
            except:
                # If all else fails, just print the equation as-is
                print(f"  {eq_name}: {eq}")
                continue
    
    # If the model has lagged variables, show them
    if hasattr(model, 'lagged_variables') and model.lagged_variables:
        print("\n=== Lagged Variables ===")
        for var in model.lagged_variables:
            print(f"  {var}")
    
    # If the model has expectations, show them
    if hasattr(model, 'expectations') and model.expectations:
        print("\n=== Expectations ===")
        for exp in model.expectations:
            print(f"  {exp}")
    
    # If the model has shocks, show them
    if hasattr(model, 'shocks') and model.shocks:
        print("\n=== Shocks ===")
        for shock in model.shocks:
            print(f"  {shock}")
    
    # Show state-space representation
    print("\n=== State-Space Representation ===")
    print("  Transition equation:")
    print("  s_t = T·s_{t-1} + R·e_t")
    print("\n  Measurement equation:")
    print("  y_t = D + Z·s_t + e_t")
    
    return model

def create_simple_model_for_display():
    """Create a simple hand-crafted DSGE model for display purposes."""
    
    # Define variables and parameters
    y = Variable('y')
    pi = Variable('pi')
    r = Variable('r')
    
    a = Parameter('alpha')
    b = Parameter('beta')
    r_nat = Parameter('r_natural')
    
    # Create simple equations
    eq1 = Equation(y, y(-1) + a*(r_nat - r(-1)), name="IS Curve")
    eq2 = Equation(pi, b*pi(-1) + (1-b)*pi(1) + a*y, name="Phillips Curve")
    eq3 = Equation(r, 1.5*pi + 0.5*y + 0.5*r(-1), name="Taylor Rule")
    
    # Create a basic model structure (simplified)
    model = {
        'variables': [y, pi, r],
        'parameters': [a, b, r_nat],
        'equations': [eq1, eq2, eq3],
        'shocks': [Variable('e_y'), Variable('e_pi'), Variable('e_r')],
        'name': 'Simple New Keynesian Model',
        'type': 'dsge'
    }
    
    return model

if __name__ == "__main__":
    # Try creating a simple hand-crafted model for display
    try:
        model = create_simple_model_for_display()
        
        # Print model details
        print("\n=== Model Details ===")
        print(f"Model Name: {model['name']}")
        print(f"Model Type: {model['type']}")
        
        print("\n=== Variables ===")
        for var in model['variables']:
            print(f"  {var}")
        
        print("\n=== Parameters ===")
        for param in model['parameters']:
            print(f"  {param}")
        
        print("\n=== Equations ===")
        for eq in model['equations']:
            print(f"  {eq}")
        
        print("\n=== Shocks ===")
        for shock in model['shocks']:
            print(f"  {shock}")
            
        print("\n=== State-Space Representation ===")
        print("  Transition equation:")
        print("  s_t = T·s_{t-1} + R·e_t")
        print("\n  Measurement equation:")
        print("  y_t = D + Z·s_t + e_t")
        
    except Exception as e:
        print(f"Error creating model for display: {e}")
        # Fall back to loading from file
        ar1_path = "dsge/examples/ar1/ar1.yaml"
        if os.path.exists(ar1_path):
            try:
                model = display_model(ar1_path)
            except Exception as e:
                print(f"Error loading AR1 model: {e}")
        else:
            print("Could not find any suitable model to display.")