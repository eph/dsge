#!/usr/bin/env python3
"""
YAML parsing utilities for DSGE models.

This module provides functions for loading and validating DSGE model specifications
from YAML files, handling different model types and validation requirements.
"""

import yaml
import re
import os
import warnings
import logging
from typing import List, Dict, Union, IO

from .DSGE import DSGE
from .FHPRepAgent import FHPRepAgent
from .SIDSGE import read_si

warnings.formatwarning = lambda message, category, filename, lineno, line=None: f'{category.__name__}: {message}\n'

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_data(data: Union[Dict, List], validator) -> None:
    """
    Validates the given data against a schema defined in the validator.

    Args:
        data (Union[Dict, List]): The data to be validated.
        validator: The validator object with a validate method and errors attribute.

    Raises:
        ValidationError: If the data fails to validate against the schema.
    """
    if not validator.validate(data):
        # Join the error messages into a single string
        error_messages = '\n'.join([f'{field}: {error}' for field, error in validator.errors.items()])
        raise ValidationError(f"Validation failed: \n{error_messages}")

def update_deprecated_keys(data):
    # Define the keys you want to replace and their new preferred names
    deprecated_keys = {
        'para_func': 'auxiliary_parameters',
        'parafunc': 'auxiliary_parameters',
        'covariances': 'covariance'
    }

    # Recursively process the data to find and replace deprecated keys
    def process_node(current_node):
        if isinstance(current_node, dict):
            for old_key, new_key in deprecated_keys.items():
                if old_key in current_node:
                    # Warn the user about the deprecated key
                    warnings.warn(f"'{old_key}' is deprecated and has been replaced with '{new_key}'. Please update your YAML files.", DeprecationWarning)

                    # Replace the old key with the new key
                    current_node[new_key] = current_node.pop(old_key)

            # Recursively process the next level of the dictionary
            for key, value in current_node.items():
                process_node(value)
        elif isinstance(current_node, list):
            # Recursively process each item in the list
            for item in current_node:
                process_node(item)

    # Start processing from the root of the data
    process_node(data)
    return data


def navigate_path(data, path):
    """Navigate through a nested dictionary using a path."""
    for key in path:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None  # Or raise an error if a key is missing
    return data

def include_constructor(loader, node):
    # Get the path to the original YAML file
    original_file = loader.name

    # Extract the directory path
    directory = os.path.dirname(original_file)

    value = loader.construct_scalar(node)
    parts = value.split('#')
    file_name = parts[0]
    fragment_path = parts[1:] if len(parts) > 1 else None

    # Construct the full path to the included file
    included_file_path = os.path.join(directory, file_name) if directory else file_name

    data = None
    # Try filesystem-relative include first
    if included_file_path and os.path.exists(included_file_path):
        with open(included_file_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        # Fallback: load include from packaged resources (e.g., dsge.schema/common.yaml)
        try:
            pkg_text = (ir_files('dsge.schema') / file_name).read_text(encoding='utf-8')
            data = yaml.safe_load(pkg_text)
        except Exception as _:
            raise FileNotFoundError(f"Included file not found: {included_file_path}")

    # Navigate the fragment path if specified
    if fragment_path:
        return navigate_path(data, fragment_path)

    return data

yaml.SafeLoader.add_constructor('!include', include_constructor)

from cerberus import Validator
# Use modern importlib.resources.files API with fallback for older Python
try:
    from importlib.resources import files as ir_files
except ImportError:  # pragma: no cover - fallback for Python <3.9
    from importlib_resources import files as ir_files

def load_schema(schema_name):
    """Load a schema YAML by name from the packaged dsge/schema directory."""
    resource_path = f"{schema_name}.yaml"
    schema_text = (ir_files('dsge.schema') / resource_path).read_text(encoding='utf-8')
    return yaml.safe_load(schema_text)

_VALIDATORS = None

def get_validators():
    global _VALIDATORS
    if _VALIDATORS is None:
        vals = {model: Validator(load_schema(model)) for model in ['fhp','lre','si']}
        vals['dsge'] = vals['lre']
        vals['sticky-information'] = vals['si']
        _VALIDATORS = vals
    return _VALIDATORS

def read_yaml(yaml_file: Union[str,IO[str]],
               sub_list : List[tuple]=[('^', '**'), (';','')]):
    """
    Read a model specification from a YAML file and return the appropriate model instance.
    
    Args:
        yaml_file: Path to a YAML file or file-like object containing the model spec
        sub_list: List of substitution patterns to apply to the YAML text
        
    Returns:
        A model instance of the appropriate type (DSGE, FHPRepAgent, SIDSGE, etc.)
        
    Raises:
        ValidationError: If the model schema validation fails
        ValueError: If model-specific validation fails
        NotImplementedError: For unsupported model types
    """
    # Configure logging for this module
    logger = logging.getLogger("dsge.parser")
    
    # Read the file content
    if isinstance(yaml_file, str):
        logger.info(f"Reading YAML from file: {yaml_file}")
        with open(yaml_file) as f:
            txt = f.read()
    else:
        logger.info("Reading YAML from stream")
        txt = yaml_file.read()

    # Apply text replacements
    for old, new in sub_list:
        txt = txt.replace(old, new)

    txt = re.sub(r"@ ?\n", " ", txt)

    # Parse YAML to dictionary
    yaml_dict = yaml.safe_load(txt)
    yaml_dict = update_deprecated_keys(yaml_dict)

    # Determine model type and validate schema
    kind = yaml_dict['declarations'].get('type', 'dsge')
    logger.info(f"Detected model type: {kind}")
    
    # Schema validation
    try:
        logger.debug("Performing schema validation")
        validate_data(yaml_dict, get_validators()[kind])
    except ValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        raise

    # Model-specific parsing and validation
    try:
        # Create appropriate model based on type
        if 'regimes' in yaml_dict:
            # Occasionally-binding constraint model
            from .obc import read_obc
            logger.debug("Creating OBC (OccBin) model with regimes/constraints")
            return read_obc(yaml_dict)
        elif kind == 'fhp':
            logger.debug("Creating FHP Representative Agent model")
            return FHPRepAgent.read(yaml_dict)
        elif kind == 'si' or kind == 'sticky-information':
            logger.debug("Creating Sticky Information DSGE model")
            return read_si(yaml_dict)
        elif kind == 'dsge-sv':
            logger.error("DSGE-SV model type not implemented")
            raise NotImplementedError('DSGE-SV model not implemented yet')
        else:
            logger.debug("Creating standard DSGE model")
            return DSGE.read(yaml_dict)
    except ValueError as e:
        # Model-specific validation errors
        logger.error(f"Model validation failed: {e}")
        raise

    
