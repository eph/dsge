#!/usr/bin/env python3
"""
Validation utilities for DSGE models.

This module provides functions for validating model specifications and detecting
common errors in model definitions.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from .symbols import Variable, Shock, Equation, TSymbol, EXP

# Configure logging
logger = logging.getLogger(__name__)


def find_symbols_in_equation(
    eq: Equation, 
    symbol_type: type, 
    names: Optional[List[str]] = None
) -> List[TSymbol]:
    """
    Find all symbols of a given type in an equation.
    
    Args:
        eq: The equation to search within
        symbol_type: The type of symbol to look for (e.g., Variable, Shock)
        names: Optional list of symbol names to restrict the search to
        
    Returns:
        List of symbols found in the equation
    """
    atoms = eq.atoms(symbol_type)
    if names:
        return [atom for atom in atoms if atom.name in names]
    return list(atoms)


def find_future_symbols(
    equations: List[Equation], 
    symbols: List[TSymbol],
    symbol_type: type = Variable
) -> Dict[Equation, List[Tuple[TSymbol, int]]]:
    """
    Find all future-dated symbols in a list of equations.
    
    Args:
        equations: List of equations to check
        symbols: List of symbols to look for
        symbol_type: Type of symbol to check (Variable or Shock)
        
    Returns:
        Dictionary mapping equations to lists of (symbol, date) tuples for future symbols
    """
    symbol_names = [s.name for s in symbols]
    result = {}
    
    for eq in equations:
        # Search both sides of the equation
        future_symbols = []
        
        for side in [eq.lhs, eq.rhs]:
            found_symbols = find_symbols_in_equation(side, symbol_type, symbol_names)
            future_symbols.extend([(s, s.date) for s in found_symbols if s.date > 0])
            
        if future_symbols:
            result[eq] = future_symbols
            
    return result


def check_for_future_shocks(
    equation_list: List[Equation], 
    shock_list: List[Variable], 
    equation_type: str,
    get_original_eq_fn: Callable[[int, str], str]
) -> None:
    """
    Check if any equation contains future-dated shocks, which are not allowed in certain models.
    
    Args:
        equation_list: List of equations to check
        shock_list: List of shock variables to look for
        equation_type: Section of the model being checked (e.g., 'cycle/terminal')
        get_original_eq_fn: Function to retrieve the original equation string
    
    Raises:
        ValueError: If any future shock is found
    """
    shock_names = [s.name for s in shock_list]
    
    for eq_idx, eq in enumerate(equation_list):
        # Find all shock instances
        future_shocks = []
        
        for side in [eq.lhs, eq.rhs]:
            symbols = find_symbols_in_equation(side, Variable, shock_names)
            future_shocks.extend([s for s in symbols if s.date > 0])
        
        if future_shocks:
            shock_names_with_dates = set(f"{s.name}({s.date})" for s in future_shocks)
            original_eq = get_original_eq_fn(eq_idx, equation_type)
            
            raise ValueError(
                f"Future shocks are not allowed in this model. Found future shock(s) "
                f"{', '.join(shock_names_with_dates)} in equation: {original_eq} "
                f"in section '{equation_type}'"
            )


def find_max_lead_lag(
    equations: List[Equation], 
    variables: List[Union[Variable, Shock]]
) -> Tuple[Dict[TSymbol, int], Dict[TSymbol, int]]:
    """
    Find the maximum lead and lag for each variable in a set of equations.
    
    Args:
        equations: List of equations to analyze
        variables: List of variables or shocks to check
        
    Returns:
        Tuple of (max_lead, max_lag) dictionaries
    """
    max_lead = {v: 0 for v in variables}
    max_lag = {v: 0 for v in variables}
    
    var_by_name = {v.name: v for v in variables}
    
    for eq in equations:
        for atom in eq.atoms():
            if isinstance(atom, Variable) and atom.name in var_by_name:
                var = var_by_name[atom.name]
                if atom.date > max_lead[var]:
                    max_lead[var] = atom.date
                elif atom.date < max_lag[var]:
                    max_lag[var] = atom.date
    
    return max_lead, max_lag


def validate_model_consistency(model_dict: Dict[str, Any]) -> List[str]:
    """
    Perform general consistency checks on a model.
    
    Args:
        model_dict: Dictionary containing model specification
        
    Returns:
        List of warning messages (empty if no issues found)
    """
    warnings = []
    
    # Check that all variables used in equations are declared
    if 'equations' in model_dict and 'variables' in model_dict:
        declared_vars = {v.name for v in model_dict['variables']}
        
        # Gather all variables used in equations
        used_vars = set()
        
        # Handle different equation formats based on model type
        equations = model_dict['equations']
        
        # Recursively process all equations regardless of structure
        def process_equations(eq_container):
            if isinstance(eq_container, list):
                # List of equations
                for eq in eq_container:
                    if hasattr(eq, 'atoms'):
                        for atom in eq.atoms(Variable):
                            used_vars.add(atom.name)
            elif isinstance(eq_container, dict):
                # Dictionary of equation sections
                for section in eq_container.values():
                    process_equations(section)
            else:
                # Skip other types
                pass
        
        # Process all equations
        process_equations(equations)
        
        # Check for undeclared variables
        undeclared = used_vars - declared_vars
        if undeclared:
            warnings.append(f"Used variables not declared: {', '.join(undeclared)}")
    
    # Similar checks could be added for shocks, parameters, etc.
    
    return warnings


def validate_dsge_leads_lags(
    equations: List[Equation], 
    variables: List[Variable],
    max_lead: int = 1, 
    max_lag: int = 1
) -> List[str]:
    """
    Validate that variables in DSGE models respect maximum lead and lag constraints.
    
    Args:
        equations: List of model equations to check
        variables: List of model variables
        max_lead: Maximum allowed lead (default: 1)
        max_lag: Maximum allowed lag (default: 1, should be positive)
        
    Returns:
        List of validation error messages (empty if no errors)
    """
    errors = []
    var_by_name = {v.name: v for v in variables}
    
    for i, eq in enumerate(equations):
        # Check for variables with leads/lags beyond limits
        for atom in eq.atoms(Variable):
            if atom.name in var_by_name:
                if atom.date > max_lead:
                    errors.append(
                        f"Variable {atom.name}({atom.date}) in equation {i+1} exceeds maximum lead of {max_lead}"
                    )
                elif atom.date < -max_lag:
                    errors.append(
                        f"Variable {atom.name}({atom.date}) in equation {i+1} exceeds maximum lag of {max_lag}"
                    )
    
    return errors


def validate_si_model(
    model_dict: Dict[str, Any],
    index_var: str
) -> List[str]:
    """
    Validate constraints specific to Sticky Information models.
    
    Args:
        model_dict: Dictionary containing model specification
        index_var: Name of the information index variable
        
    Returns:
        List of validation error messages (empty if no errors)
    """
    errors = []
    
    # Ensure index variable appears somewhere in equations (as Variable, Parameter, or symbol)
    if 'equations' in model_dict:
        index_found = False
        for eq in model_dict['equations']:
            # Look for a Variable named index_var
            if any(getattr(atom, 'name', None) == index_var for atom in eq.atoms(Variable)):
                index_found = True
                break
            # Or a Parameter named index_var
            from .symbols import Parameter as _Param
            if any(getattr(atom, 'name', None) == index_var for atom in eq.atoms(_Param)):
                index_found = True
                break
            # Or a raw sympy Symbol named index_var (in case it was parsed that way)
            import sympy as _sp
            free_syms = getattr(eq, 'free_symbols', set()) or set()
            if any(str(sym) == index_var for sym in free_syms):
                index_found = True
                break
        if not index_found:
            errors.append(f"Index variable {index_var} not found in model equations")
    
    return errors
