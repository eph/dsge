# LINVER (FRB/US linearized) examples + translation checks

The Federal Reserve publishes the LINVER (linearized FRB/US) model and tooling here:
`https://www.federalreserve.gov/econres/us-models-linver.htm`.

The download includes Octave/Matlab code that generates one or more Dynare `.mod` files. This repo
keeps a YAML translation (`mcap.yaml`) and provides scripts to:

- import `.mod` files to YAML, and
- verify that our YAML matches the generated `.mod`.

## Workflow

1. Download and unzip the Fed LINVER package.
2. Run the Octave/Matlab script(s) that generate the `.mod` files (per the Fed instructions).
3. Use the scripts below.

## Import `.mod` → YAML

```bash
.venv/bin/python dsge/examples/linver/import_linver_mods.py /path/to/generated_mods_dir --out dsge/examples/linver/_imported
```

This writes YAML files that should be readable by `dsge.read_yaml(...)`.

## Verify translation (YAML vs `.mod`)

```bash
.venv/bin/python dsge/examples/linver/verify_linver_translation.py \\
  --yaml dsge/examples/linver/mcap.yaml \\
  --mod  /path/to/mcap.mod
```

The verifier does:
- set comparisons of variables/shocks/parameters,
- calibration diffs (for common parameters), and
- a randomized numeric “equation equivalence” check equation-by-equation (fast, avoids heavy SymPy simplification).

## Variants (MCAP+WP, VAR-based, …)

If the Fed package generates multiple `.mod` variants, point `import_linver_mods.py` at the directory
containing them; it will import all `*.mod` files it finds. Once we confirm the mappings, we can
commit the additional YAML variants (e.g. `mcap_wp.yaml`, `mcap_var.yaml`) alongside `mcap.yaml`.

