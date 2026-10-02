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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Dict, Optional, Union, IO

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

class _DSGELoader(yaml.SafeLoader):
    """Private loader: importing DSGE never changes PyYAML's global loaders."""


def include_constructor(loader, node):
    value = loader.construct_scalar(node)
    if loader.include_resolver is None:
        raise ValueError("YAML !include is disabled; supply an explicit include_root")
    return loader.include_resolver(value)


_DSGELoader.add_constructor('!include', include_constructor)


def _include_parts(value):
    file_name, *fragment = value.split('#')
    path = PurePosixPath(file_name)
    if (not file_name or path.is_absolute() or PureWindowsPath(file_name).drive
            or ".." in path.parts or "\\" in file_name):
        raise ValueError("Include paths must be relative and cannot contain parent traversal")
    return path, fragment


def _load_document(text, resolver=None):
    loader = _DSGELoader(text)
    loader.include_resolver = resolver
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def load_yaml(text, *, include_root=None, source_path=None,
              max_include_depth=16, max_include_bytes=10 * 1024 * 1024):
    """Load YAML, optionally allowing bounded includes inside a caller-owned root.

    Includes are disabled by default. Relative names resolve against the source
    document's directory (or include_root for a stream). Absolute paths, parent
    traversal, symlink escapes, cycles, and budget overruns are rejected.
    """
    if max_include_depth < 0 or max_include_bytes < 1:
        raise ValueError("Include limits must be nonnegative depth and positive bytes")
    root = Path(include_root).resolve(strict=True) if include_root is not None else None
    if root is not None and not root.is_dir():
        raise ValueError("include_root must be a directory")
    source = Path(source_path).resolve() if source_path is not None else None
    remaining = max_include_bytes - len(text.encode("utf-8"))
    if remaining < 0:
        raise ValueError("YAML exceeds max_include_bytes")
    active = {source} if source is not None else set()

    def parse(document, directory, depth):
        def include(value):
            nonlocal remaining
            if depth >= max_include_depth:
                raise ValueError("YAML include depth limit exceeded")
            path, fragment = _include_parts(value)
            target = (directory / path).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError("YAML include escapes include_root")
            if target in active:
                raise ValueError("YAML include cycle detected")
            if not target.is_file():
                raise ValueError("YAML includes must refer to regular files")
            with target.open("rb") as stream:
                content = stream.read(remaining + 1)
            remaining -= len(content)
            if remaining < 0:
                raise ValueError("YAML include byte limit exceeded")
            active.add(target)
            try:
                result = parse(content.decode("utf-8"), target.parent, depth + 1)
            finally:
                active.remove(target)
            return navigate_path(result, fragment) if fragment else result
        return _load_document(document, include if root is not None else None)

    return parse(text, source.parent if source is not None else root, 0)


from cerberus import Validator
# Use modern importlib.resources.files API with fallback for older Python
try:
    from importlib.resources import files as ir_files
except ImportError:  # pragma: no cover - fallback for Python <3.9
    from importlib_resources import files as ir_files

def load_schema(schema_name):
    """Load trusted schema includes exclusively from packaged resources."""
    resource_root = ir_files('dsge.schema')
    active = set()

    def load(path):
        name = path.as_posix()
        if name in active or len(active) >= 16:
            raise ValueError("Schema include cycle or depth limit exceeded")
        active.add(name)
        try:
            text = resource_root.joinpath(*path.parts).read_text(encoding='utf-8')

            def include(value):
                child, fragment = _include_parts(value)
                result = load(path.parent / child)
                return navigate_path(result, fragment) if fragment else result

            return _load_document(text, include)
        finally:
            active.remove(name)

    path, _ = _include_parts(f"{schema_name}.yaml")
    return load(path)

_VALIDATORS = None

def get_validators():
    global _VALIDATORS
    if _VALIDATORS is None:
        vals = {model: Validator(load_schema(model)) for model in ['fhp', 'lre', 'si', 'switching_ssm']}
        vals['dsge'] = vals['lre']
        vals['sticky-information'] = vals['si']
        _VALIDATORS = vals
    return _VALIDATORS

def read_yaml(yaml_file: Union[str,IO[str]],
               sub_list : List[tuple]=[('^', '**'), (';','')], *,
               include_root=None, max_include_depth=16,
               max_include_bytes=10 * 1024 * 1024):
    """
    Read a model specification from a YAML file and return the appropriate model instance.
    
    Args:
        yaml_file: Path to a YAML file or file-like object containing the model spec
        sub_list: List of substitution patterns to apply to the YAML text
        include_root: Explicit trusted root enabling relative YAML includes
        max_include_depth: Maximum nesting of enabled includes
        max_include_bytes: Maximum total UTF-8 bytes across YAML and includes
        
    Returns:
        A model instance of the appropriate type (DSGE, FHPRepAgent, SIDSGE, etc.)
        
    Raises:
        ValidationError: If the model schema validation fails
        ValueError: If model-specific validation fails
        NotImplementedError: For unsupported model types
    """
    # Configure logging for this module
    logger = logging.getLogger("dsge.parser")
    
    # Bound root input as well as included files before allocating the text.
    if not isinstance(max_include_bytes, int) or max_include_bytes < 1:
        raise ValueError("max_include_bytes must be a positive integer")
    if not isinstance(max_include_depth, int) or max_include_depth < 0:
        raise ValueError("max_include_depth must be a nonnegative integer")
    if isinstance(yaml_file, str):
        logger.info(f"Reading YAML from file: {yaml_file}")
        yaml_path = os.path.abspath(yaml_file)
        yaml_dir = os.path.dirname(yaml_path)
        with open(yaml_file, "rb") as f:
            raw = f.read(max_include_bytes + 1)
        if len(raw) > max_include_bytes:
            raise ValueError("YAML exceeds max_include_bytes")
        txt = raw.decode("utf-8")
    else:
        logger.info("Reading YAML from stream")
        yaml_path = None
        yaml_dir = None
        txt = yaml_file.read(max_include_bytes + 1)
        if len(txt.encode("utf-8")) > max_include_bytes:
            raise ValueError("YAML exceeds max_include_bytes")

    # Apply text replacements
    for old, new in sub_list:
        txt = txt.replace(old, new)

    txt = re.sub(r"@ ?\n", " ", txt)

    # Parse YAML to dictionary
    yaml_dict = load_yaml(
        txt, include_root=include_root, source_path=yaml_path,
        max_include_depth=max_include_depth, max_include_bytes=max_include_bytes,
    )
    yaml_dict = update_deprecated_keys(yaml_dict)

    def _find_project_root(start_dir: str) -> Optional[str]:
        """
        Heuristic: walk upward to find a repo/project root (pyproject.toml or .git).
        """
        cur = os.path.abspath(start_dir)
        for _ in range(50):
            if os.path.exists(os.path.join(cur, "pyproject.toml")) or os.path.exists(os.path.join(cur, ".git")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    # Resolve relative file references against the YAML directory (when available).
    if yaml_dir and isinstance(yaml_dict, dict):
        project_root = _find_project_root(yaml_dir)
        est = yaml_dict.get("estimation", {})
        if isinstance(est, dict) and "data" in est:
            data_spec = est.get("data")
            if isinstance(data_spec, dict) and "file" in data_spec:
                f = data_spec.get("file")
                if isinstance(f, str) and not os.path.isabs(f):
                    candidate = os.path.join(yaml_dir, f)
                    if os.path.exists(candidate):
                        data_spec["file"] = candidate
                    elif project_root:
                        candidate = os.path.join(project_root, f)
                        if os.path.exists(candidate):
                            data_spec["file"] = candidate
            elif isinstance(data_spec, str) and not os.path.isabs(data_spec):
                candidate = os.path.join(yaml_dir, data_spec)
                if os.path.exists(candidate):
                    est["data"] = candidate
                elif project_root:
                    candidate = os.path.join(project_root, data_spec)
                    if os.path.exists(candidate):
                        est["data"] = candidate

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
            model_obj = read_obc(yaml_dict)
        elif kind == 'fhp':
            logger.debug("Creating FHP Representative Agent model")
            model_obj = FHPRepAgent.read(yaml_dict)
            # Optional: endogenous horizon switching when declarations.stopping_rule / declarations.horizon_choice is present.
            dec = yaml_dict.get("declarations")
            if isinstance(dec, dict) and ("horizon_choice" in dec or "stopping_rule" in dec):
                model_obj = model_obj.compile_endogenous_horizon_model()
        elif kind == "switching_ssm":
            logger.debug("Creating switching state-space model")
            from .switching_ssm import read_switching_ssm

            warnings.warn(
                "YAML type 'switching_ssm' is experimental and may be removed in a future release. "
                "Prefer FHP models with declarations.stopping_rule for endogenous horizons when possible.",
                DeprecationWarning,
            )
            model_obj = read_switching_ssm(yaml_dict)
        elif kind == 'si' or kind == 'sticky-information':
            logger.debug("Creating Sticky Information DSGE model")
            model_obj = read_si(yaml_dict)
        elif kind == 'dsge-sv':
            logger.error("DSGE-SV model type not implemented")
            raise NotImplementedError('DSGE-SV model not implemented yet')
        else:
            logger.debug("Creating standard DSGE model")
            model_obj = DSGE.read(yaml_dict)
    except ValueError as e:
        # Model-specific validation errors
        logger.error(f"Model validation failed: {e}")
        raise

    # Attach source metadata for downstream tools (e.g., Slurm backends).
    if yaml_path is not None:
        try:
            setattr(model_obj, "source_file", yaml_path)
            setattr(model_obj, "source_dir", yaml_dir)
        except Exception:
            pass

    return model_obj

    
