"""Historical task probes for the 2026-10-02 audit, not regression tests.

Usage: python audit-2026-10-02.py CHECKOUT OUTPUT_JSON [TASK_ID ...]
Use a disposable checkout at one of the audited revisions. Expected unresolved
behavior is recorded as data; a probe_error means the audit itself failed.
Generated C++ exports use a temporary directory. No tracker is required.
"""
from pathlib import Path
import contextlib
import io
import inspect
import json
import os
import sys
import tempfile
import traceback

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
import numpy as np
import yaml
from dsge import read_yaml
from dsge.irfoc import IRFOC, IRFOCResult

selected = sys.argv[3:]
results = json.loads(output.read_text()) if selected and output.exists() else {'source_root': str(root), 'probes': {}}

def probe(task, fn):
    if selected and task not in selected:
        return
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            result = fn()
    except Exception:
        result = {'probe_error': traceback.format_exc()}
    results['probes'][task] = result
    print(task, json.dumps(result), flush=True)
    output.write_text(json.dumps(results, indent=2)+'\n')

def cpp_paths():
    from dsge.translate_cpp import translate_cpp
    model = read_yaml(str(root/'dsge/examples/fhp/fhp.yaml'))
    with tempfile.TemporaryDirectory(prefix='dsge-audit-cpp-') as td:
        translate_cpp(model, output_dir=td)
        lines = [s for s in (Path(td)/'CMakeLists.txt').read_text().splitlines() if 'include_directories' in s]
    return {'generated_include_directories': lines}


def anticipated_zero():
    model = read_yaml(io.StringIO('''
declarations:
  name: audit_ar1
  variables: [x]
  parameters: [rho]
  shocks: [e]
equations:
  - x = rho*x(-1) + e
calibration:
  parameters: {rho: 0.85}
'''))
    compiled = model.compile_model()
    ordinary = compiled.impulse_response(model.p0(), h=5)['e']['x'].tolist()
    try:
        got = compiled.anticipated_impulse_response(model.p0(), anticipated_h=0, h=5)
        return {'ordinary': ordinary, 'anticipated_zero': got['e']['x'].tolist()}
    except Exception as e:
        return {'ordinary': ordinary, 'anticipated_zero_error': f'{type(e).__name__}: {e}'}


def irfoc_api():
    return {'constructor': str(inspect.signature(IRFOC)),
            'simulate': str(inspect.signature(IRFOC.simulate)),
            'public_methods': [name for name, fn in inspect.getmembers(IRFOC, callable) if not name.startswith('_')],
            'result_fields': list(IRFOCResult.__dataclass_fields__),
            'horizon_from_baseline': 'self.T, self.n = self.baseline.shape' in inspect.getsource(IRFOC.__init__)}


def owner_forecasts():
    spec = yaml.safe_load((root/'dsge/examples/fhp/nk_endogenous_own_horizon_projection.yaml').read_text())
    spec['declarations']['expectations'] = 1
    try:
        read_yaml(io.StringIO(yaml.safe_dump(spec, sort_keys=False)))
        return {'accepted_expectations': True}
    except Exception as e:
        return {'expectations_error': f'{type(e).__name__}: {e}'}


def dgs_path():
    path = root/'dsge/examples/DGS/DGS.yaml'
    model = read_yaml(str(path))
    spec = model['__data__']
    kwargs = {'allow_external_code': True} if 'allow_external_code' in inspect.signature(model.compile_model).parameters else {}
    result = {'declared_helper': spec['declarations']['external']['file'], 'compile_options': kwargs}
    try:
        model.compile_model(**kwargs)
        result['compile'] = 'success'
    except Exception as e:
        result['compile_error'] = f'{type(e).__name__}: {e}'
    return result


def uneven_horizons():
    spec = yaml.safe_load((root/'dsge/examples/fhp/nk_endogenous_exponential_costs.yaml').read_text())
    for component in spec['declarations']['stopping_rule']['components'].values():
        component['k_max'] = 370
    model = read_yaml(io.StringIO(yaml.safe_dump(spec, sort_keys=False)))
    state = np.array([0.0025])
    regime = model.choose_regime(model.p0, state, t=0)
    _, _, zz, dd, _, _ = model.get_mats(model.p0, regime)
    actual = zz @ state + dd
    values = dict(zip(model.parameter_names, model.p0))
    beta, sigma, kappa, rho = (values[name] for name in ('beta','sigma','kappa','rho_d'))
    D = np.linalg.inv(np.array([[1.,0.],[-kappa,1.]]))
    MH = D @ np.array([[1.,sigma],[0.,0.]])
    MF = D @ np.array([[0.,0.],[0.,beta]])
    matrix = MH + MF
    b = D @ np.array([sigma,0.])
    def common(h):
        total = np.zeros(2)
        increment = state[0]*b
        for _ in range(h):
            total += increment
            increment = rho*matrix @ increment
        return total
    h, f = regime
    exact = state[0]*b + rho*MH @ common(h) + rho*MF @ common(f)
    return {'regime': list(regime), 'actual': actual.tolist(), 'independent_recursion': exact.tolist(),
            'absolute_output_error': float(abs(actual[0]-exact[0]))}


def prehistory():
    from dsge.tests.test_irfoc import _simple_nk_yaml
    model = read_yaml(io.StringIO(_simple_nk_yaml()))
    compiled = model.compile_model()
    baseline = compiled.impulse_response(model.p0(), h=20)['er'][['pi','y','i','ilag']]
    offset = 5
    historical_pi = float(baseline.iloc[offset-1]['pi'])
    shifted = baseline.iloc[offset:].reset_index(drop=True)
    irfoc = IRFOC(model, shifted, 'em', p0=model.p0(), compiled_model=compiled)
    result = irfoc.simulate('i = 0.5*pi(-1)', return_details=True)
    return {'prior_pi': historical_pi, 'expected_i0_with_presample_history': 0.5*historical_pi,
            'actual_i0': float(result.simulation.iloc[0]['i']),
            'reported_residual0': result.residuals.iloc[0].tolist(),
            'has_pi_lag_alias': ('pi',-1) in irfoc._time_shift_alias_cols}

probe('dsge-0qi', cpp_paths)
probe('dsge-5fc', anticipated_zero)
probe('dsge-845+dsge-y4d', irfoc_api)
probe('dsge-846', owner_forecasts)
probe('dsge-847', dgs_path)
probe('dsge-850', uneven_horizons)
probe('dsge-gwm', prehistory)
