"""Host-controlled resolution of numerical functions declared by a model."""

from collections.abc import Mapping
from importlib.machinery import SourceFileLoader
import importlib.util
import keyword
from pathlib import Path


def external_function_key(model, *, external_functions=None, allow_external_code=False):
    """Validate authority without importing a file, and identify the numerical context.

    The registry is supplied by the caller, never by model metadata. File plugins
    require a fresh explicit opt-in on every compilation request, including cache hits.
    """
    external = model.get("__data__", {}).get("declarations", {}).get("external")
    if external is None:
        return ("none",)
    if not isinstance(external, Mapping):
        raise ValueError("declarations.external must be a mapping.")
    names = external.get("names")
    if not isinstance(names, (list, tuple)) or any(
        not isinstance(name, str)
        or not name.isidentifier()
        or "__" in name
        or keyword.iskeyword(name)
        for name in names
    ):
        raise ValueError("declarations.external.names must be a list of function names.")
    if external_functions is not None:
        if not isinstance(external_functions, Mapping):
            raise TypeError("external_functions must be a name-to-callable mapping.")
        for name in names:
            if name not in external_functions or not callable(external_functions[name]):
                raise ValueError(f"external_functions must supply a callable for {name!r}.")
        return ("registry", tuple((name, id(external_functions[name])) for name in names))
    if allow_external_code is not True:
        raise PermissionError(
            "Model-declared Python files are disabled. Supply an external_functions "
            "registry, or pass allow_external_code=True only for trusted model bundles; "
            "this executes the entire Python file with your process permissions."
        )
    filename = external.get("file")
    if not isinstance(filename, (str, Path)) or not str(filename):
        raise ValueError("declarations.external.file must name a Python file.")
    return ("file", str(filename), tuple(names))


def resolve_external_functions(model, *, external_functions=None, allow_external_code=False):
    """Return declared callables, importing model-selected Python only with host consent.

    A supplied registry takes precedence and never opens the model's file. Opting
    into file plugins grants arbitrary Python execution; it is not a sandbox.
    """
    key = external_function_key(
        model,
        external_functions=external_functions,
        allow_external_code=allow_external_code,
    )
    if key[0] == "none":
        return {}
    if key[0] == "registry":
        return {name: external_functions[name] for name, _ in key[1]}

    _, filename, names = key
    spec = importlib.util.spec_from_loader("external", SourceFileLoader("external", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = {}
    for name in names:
        function = getattr(module, name)
        if not callable(function):
            raise ValueError(f"External function {name!r} is not callable.")
        functions[name] = function
    return functions
