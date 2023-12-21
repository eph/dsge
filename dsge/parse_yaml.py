#!/usr/bin/env python3
import yaml, re, os, warnings
from typing import List, Dict, Union

from .DSGE import DSGE
from .FHPRepAgent import FHPRepAgent

warnings.formatwarning = lambda message, category, filename, lineno, line=None: f'{category.__name__}: {message}\n'

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_data(data, validator):
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
    included_file_path = os.path.join(directory, file_name)

    with open(included_file_path, 'r') as f:
        data = yaml.safe_load(f)

    # Navigate the fragment path if specified
    if fragment_path:
        return navigate_path(data, fragment_path)

    return data

yaml.SafeLoader.add_constructor('!include', include_constructor)

from cerberus import Validator
import importlib.resources as pkg_resources

def load_schema(schema_name):
    # Use the package name and the relative path to the schema file
    resource_path = f'{schema_name}.yaml'

    # Open the resource within the package context
    with pkg_resources.open_text('dsge.schema', resource_path) as f:
        schema = yaml.safe_load(f)
    return schema

# Example usage
validators = {model: Validator(load_schema(model)) for model in ['fhp','lre']}
validators['dsge'] = validators['lre']

def read_yaml(yaml_file: str,
               sub_list : List[tuple]=[('^', '**'), (';','')]):
    """
    This function reads a yaml file and returns a dictionary.
    """
    with open(yaml_file) as f:
        txt = f.read()

    for old, new in sub_list:
        txt = txt.replace(old, new)

    txt = re.sub(r"@ ?\n", " ", txt)

    yaml_dict = yaml.safe_load(txt)

    yaml_dict = update_deprecated_keys(yaml_dict)

    kind =  yaml_dict['declarations'].get('type','dsge')
    
    try:
        validate_data(yaml_dict, validators[kind])
    except ValidationError as e:
        print(e)

    if kind=='fhp':
        return FHPRepAgent.read(yaml_dict)
    elif kind=='si':
        raise NotImplementedError('SI model not implemented yet')
    elif kind=='dsge-sv':
        raise NotImplementedError('DSGE-SV model not implemented yet')
    else:
        return DSGE.read(yaml_dict)

    
