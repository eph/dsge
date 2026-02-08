# LINVER (FRB/US linearized) examples + translation checks

The Federal Reserve publishes the LINVER (linearized FRB/US) model and tooling here:
`https://www.federalreserve.gov/econres/us-models-linver.htm`.

The download includes Octave/Matlab code that generates one or more Dynare `.mod` files. This repo
keeps a YAML translation (`mcap.yaml`) and provides scripts to:

- import `.mod` files to YAML, and
- verify that our YAML matches the generated `.mod`.

## Workflow

1. Download and unzip the Fed LINVER package.
2. Generate `runmod.mod` using the Fed `make_runmod.m` (Matlab) or `make_runmod_octave.m` (Octave).
3. Use the scripts below to import/verify.

### Generate `runmod.mod` (Octave)

The Fed package’s `make_runmod_octave.m` ends by calling Dynare to *parse* the model. For translation
checks we only need the generated `runmod.mod`, so this repo includes a small wrapper that installs a
temporary `dynare.m` stub (unless you pass `--use-dynare`).

```bash
.venv/bin/python dsge/examples/linver/generate_runmod_mods.py /path/to/public_linver \\
  --out /tmp/linver_mods \\
  --expvers mcap mcapwp var \\
  --mprule intay
```

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

This repo includes the translated variants:

- `mcap.yaml` (MCAP expectations, inertial Taylor rule)
- `mcapwp.yaml` (MCAP+WP expectations, inertial Taylor rule)
- `var.yaml` (VAR expectations, inertial Taylor rule)

To regenerate, use `generate_runmod_mods.py` to create the `.mod` files and then import them with
`import_linver_mods.py`.
