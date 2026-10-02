# Expression Parsing DSL

DSGE constructs SymPy expressions from a restricted mathematical grammar. It does
not evaluate model strings as Python. The same parser is used for model equations,
policy rules, losses, fixed-parameter formulas, and transition matrices.

- Declare variables, parameters, shocks, and auxiliary parameters in the model.
  Unknown identifiers are rejected. Names must be valid identifiers without
  double underscores.
- Constants `pi`, `E`, `I`, `oo`, and `inf` have their SymPy meanings unless a
  declared model symbol overrides the name.
- Use numeric literals and `+`, `-`, `*`, `/`, and `**`. `^` is accepted for powers.
- Use integer lead/lag offsets: `x(+1)`, `x(-1)`, and `x` for the current period.
- Use `EXP(j)(expression)` for expectations and `SUM(expression, (j, a, b))`
  for sums. Declare index symbols such as `j`; `oo` denotes infinity.
- Mathematical functions include `exp`, `log`, trigonometric and hyperbolic
  functions, `sqrt`, `Abs`, `sign`, `Min`, `Max`, and `Piecewise`.
- Relations and Boolean combinations are available for constraints and policies.
  Matrices use rectangular nested lists or `Matrix([[...]])`.
- DataFrame variable labels must name a declared variable, optionally with an
  integer lead/lag. They cannot contain general formulas.

Python attribute access, indexing, string literals, comprehensions, lambdas,
imports, and arbitrary calls are rejected before symbolic construction. Python
`sum`/`range` expressions must be rewritten as explicit arithmetic or DSL `SUM`.
Host applications should still impose their own process resource limits when
processing large or untrusted symbolic models.

## External functions

A model may declare symbolic external function names, but model-selected Python
files are disabled during compilation by default. Prefer supplying a trusted
`external_functions` mapping from names to callables in the host application.
For an audited model and companion file, `allow_external_code=True` explicitly
permits the file to run with the Python process's full authority. This option is
not a sandbox. The first- and second-order compilation paths apply the same trust
policy; a model cannot grant itself permission in YAML. These options are
available on `compile_model`, `python_sims_matrices`, and `solve_second_order`.
Each direct compilation request needs its own opt-in; a compiled second-order
object retains a copy of the caller's registry and trust options.

For second-order compilation, `max_tensor_entries` defaults to 2,000,000 and is
checked before symbolic derivatives are constructed. The compiled object retains
this limit for likelihood and simulation; direct solves also enforce it on cache
hits.

## YAML includes

`read_yaml` disables `!include` by default. To enable includes, the caller must
supply an explicit directory containing the permitted files:

```python
model = read_yaml("models/main.yaml", include_root="models")
```

Relative names resolve against the including file's directory. Absolute paths,
parent traversal, symlink escapes, and cycles are rejected. The optional
`max_include_depth` (default 16) and `max_include_bytes` (default 10 MiB) bound
include nesting and total UTF-8 input bytes. Includes in a stream resolve against
the explicit root. `load_yaml` exposes the same policy when a plain dictionary is
needed, with an optional `source_path` for relative resolution.

Internal schemas resolve includes only within packaged schema resources.
Importing DSGE does not add constructors to `yaml.SafeLoader` or change unrelated
`yaml.safe_load` calls.

## Packaged resources

Use `resource_path` for scoped filesystem access to package files.
`resource_file_path` provides a path valid until process exit: source installations
use the packaged file directly, and zip installations use private temporary
files managed by `importlib.resources`. No shared persistent resource cache is
trusted. Resource paths must remain relative to the package and cannot traverse
parent directories.

## Generated source

Fortran and Dynare exporters validate identifiers as ASCII names beginning with
letters, with at most 63 characters. Model display names may contain printable
text but not control characters. Fortran string literals are encoded, and Dynare
expressions are serialized from supported mathematical nodes rather than copied
as native source. Unsupported directives and source fragments are rejected.

`run_dynare_mod_text` is a low-level API for executing trusted Dynare source.
Likewise, explicitly supplied Fortran `extra_code` and `filter_function` hooks
are trusted caller code. Keep these separate from untrusted model metadata.
