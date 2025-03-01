#!/usr/bin/env python3
"""
Test the enhanced string representation of DSGE symbols.
"""

import sympy
from dsge.symbols import Variable, Parameter, Shock

def print_symbol_examples():
    """Print examples of the new symbol representation."""
    print("Parameter examples:")
    params = [
        Parameter("alpha"),
        Parameter("beta"),
        Parameter("rho"),
        Parameter("sigma"),
        Parameter("gamma"),
        Parameter("phi"),
        Parameter("theta"),
        Parameter("rho_z"),   # Parameter with suffix
        Parameter("non_greek")  # Non-Greek parameter
    ]
    
    for p in params:
        print(f"  {p.name} → {p}")
    
    print("\nVariable examples (current period):")
    vars_current = [
        Variable("y"),        # Standard variable
        Variable("pi"),       # Greek-named variable (inflation)
        Variable("lambda"),   # Lambda variable
        Variable("theta")     # Theta variable
    ]
    
    for v in vars_current:
        print(f"  {v.name} → {v}")
    
    print("\nVariable examples (with leads/lags):")
    y = Variable("y")
    pi = Variable("pi")
    examples = [
        y(-1),      # y lagged 1 period
        y(-2),      # y lagged 2 periods
        y(1),       # y led 1 period
        y(2),       # y led 2 periods
        pi(-1),     # pi lagged 1 period
        pi(1)       # pi led 1 period
    ]
    
    for ex in examples:
        print(f"  {ex.name}({ex.date}) → {ex}")
    
    print("\nVariable examples with expectations:")
    # Create variables with expectations
    from sympy import symbols
    y_exp = Variable("y", exp_date=1)
    pi_exp = Variable("pi", exp_date=1)
    c_lead_exp = Variable("c", date=1, exp_date=1)  # Expectation of future c
    
    print(f"  E_t[y_t] → {y_exp}")
    print(f"  E_t[π_t] → {pi_exp}")
    print(f"  E_t[c_{{t+1}}] → {c_lead_exp}")
    
    # Create lagged expectations examples
    from dsge.symbols import LaggedExpectation
    
    # Expectation formed in the past (t-1) of current period value
    y_lag_exp = LaggedExpectation("y", date=0, exp_date=1)
    # Expectation formed in the past (t-2) of future (t+1) value
    c_lag_lead_exp = LaggedExpectation("c", date=1, exp_date=2)
    
    print("\nLagged expectation examples:")
    print(f"  E_{{t-1}}[y_t] → {y_lag_exp}")
    print(f"  E_{{t-2}}[c_{{t+1}}] → {c_lag_lead_exp}")
    
    print("\nEquation examples:")
    y = Variable("y")
    c = Variable("c")
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    
    # Import Equation class and create actual equation objects
    from dsge.symbols import Equation
    
    # Simple equation: y = c + alpha*beta
    eq1 = Equation(y, c + alpha*beta, name="Output")
    print(f"  With name: {eq1}")
    print(f"  Without name: {Equation(y, c + alpha*beta)}")
    
    # Direct string substitution equation for comparison
    direct_eq = f"{y} = {c} + {alpha}·{beta}"
    print(f"  Direct substitution: {direct_eq}")
    
    # Examples of more complex equations
    print("\nMore complex equation examples:")
    
    # Equation with leads and lags: y = beta*y(-1) + alpha*c(+1)
    eq2 = Equation(y, beta*y(-1) + alpha*c(1), name="Dynamic output")
    print(f"  {eq2}")
    
    # Equation with expectations: y = beta*E_t[y_{t+1}] + alpha*c
    y_exp = Variable("y", date=1, exp_date=1)  # E_t[y_{t+1}]
    eq3 = Equation(y, beta*y_exp + alpha*c, name="Forward-looking")
    print(f"  {eq3}")
    
    # New Keynesian Phillips Curve example
    pi = Variable("pi")  # Inflation
    pi_exp = Variable("pi", date=1, exp_date=1)  # Expected inflation
    y_gap = Variable("y_gap")  # Output gap
    kappa = Parameter("kappa")  # Slope parameter
    
    nkpc = Equation(pi, beta*pi_exp + kappa*y_gap, name="NKPC")
    print(f"  {nkpc}")
    
    # Example with a squared term
    sigma = Parameter("sigma")  # Standard deviation
    variance = sympy.Symbol("var")  # Variance
    var_eq = Equation(variance, sigma**2, name="Variance")
    print(f"  {var_eq}")
    
    # Interest rate rule with multiple components (Taylor rule)
    r = Variable("r")  # Interest rate
    r_target = Variable("r_bar")  # Target interest rate
    pi_gap = Variable("pi") - Variable("pi_bar")  # Inflation gap
    y_gap = Variable("y_gap")  # Output gap
    phi_pi = Parameter("phi_pi")  # Weight on inflation
    phi_y = Parameter("phi_y")  # Weight on output
    rho = Parameter("rho")  # Smoothing parameter
    
    taylor = Equation(r, rho*r(-1) + (1-rho)*(r_target + phi_pi*pi_gap + phi_y*y_gap), name="Taylor rule")
    print(f"  {taylor}")
    
    # Higher order terms
    x = Variable("x")
    cubic = Equation(x, x(-1)**3 + rho*x(-2), name="Cubic AR(2)")
    print(f"  {cubic}")

if __name__ == "__main__":
    print_symbol_examples()