import io

from dsge.dynare_export import to_dynare_mod
from dsge.parse_yaml import read_yaml


def test_to_dynare_mod_smoke_contains_blocks():
    yaml_text = """
declarations:
  name: dynare_export_smoke
  variables: [x, y]
  shocks: [e]
  parameters: [a, b]

equations:
  model:
    - x = a*x(-1) + b*e
    - y = x + y(1)
  observables:
    x: x
    y: y

calibration:
  parameters:
    a: 0.9
    b: 1.0
  covariance:
    e: 0.01
"""
    m = read_yaml(io.StringIO(yaml_text))
    mod = to_dynare_mod(m, order=2, pruning=True)
    txt = mod.mod_text

    assert "parameters a b;" in txt
    assert "var x y;" in txt
    assert "varexo e;" in txt
    assert "model;" in txt and "end;" in txt
    assert "shocks;" in txt and "stderr sqrt(0.01" in txt
    assert "stoch_simul(order=2, pruning, irf=0);" in txt
