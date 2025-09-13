# Expression Parsing DSL

This project parses model equations from YAML into SymPy expressions using a safe, whitelisted context. This page summarizes the supported syntax and functions.

- Declared symbols
  - Variables: declared under `declarations.variables` (e.g., `y`, `pinf`).
  - Parameters: declared under `declarations.parameters`.
  - Shocks/Innovations: declared under `declarations.shocks` (and `innovations` for FHP models).
  - All symbols used in equations must be declared; unknown names raise a parsing error.

- Leads/Lags
  - Use parentheses with integer offsets: `x(+1)`, `x(-1)`, `x` for current period.
  - Offsets apply to both endogenous variables and shocks.

- Expectations
  - Use `EXP(j)(expr)` to denote an expectation formed at time `t-j` of `expr`.
  - Example: `SUM((1-lam)**j * EXP(-j-1)(pp + alp*ygr), (j, 0, oo))`.

- Sums and indexing
  - Use `SUM(expr, (i, a, b))` for summation, with `oo` for infinity (SymPy’s `oo`).
  - Index symbols (like `j` above) must be declared as parameters.

- Math functions
  - Supported: `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sqrt`, `Abs`, `sign`.
  - Add custom functions via `declarations.external` (DSGE) or code paths that load externals.

- Operators
  - Standard Python/SymPy operators are supported (e.g., `+`, `-`, `*`, `/`).
  - Power: `**` is preferred; `^` is accepted in existing examples for exponents.

- Observables
  - Observable equations are parsed in the same controlled context as model equations.
  - Common Python helpers like `range`/`sum` are available where used.

- Reserved names and collisions
  - The parser avoids injecting SymPy’s `pi` constant to prevent conflicts with variables named `pi` or `pinf`.

- Validation and errors
  - Unknown symbols trigger a clear `ValueError` during parsing.
  - Leads/lags and other model constraints are validated during model construction.

If you encounter an expression that is not covered by this DSL, consider either using an equivalent SymPy-friendly form or extending the parsing context in code.

