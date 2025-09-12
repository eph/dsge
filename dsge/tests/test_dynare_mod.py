from pathlib import Path
import tempfile

import numpy as np
from numpy.testing import assert_allclose

from dsge.dynare_parser import parse_mod_text
from dsge.read_mod import read_mod


SIMPLE_MOD = """
// Simple toy model
parameters beta rho sigma;
beta = 0.99; rho = 0.9; sigma = 0.1;

var y;
varexo e;

model;
  y = beta*y(+1) + e;
end;

shocks; var e; stderr 0.1; end;

initval; y = 0; end;
"""


def test_parse_mod_text_basic():
    parsed = parse_mod_text(SIMPLE_MOD)
    assert set(parsed["parameters"]) == {"beta", "rho", "sigma"}
    assert set(parsed["variables"]) == {"y"}
    assert set(parsed["shocks"]) == {"e"}
    assert any(eq.strip().startswith("y =") for eq in parsed["equations"])  # has equation
    # covariance mapped to variance
    assert "e" in parsed["covariance"]
    assert parsed["covariance"]["e"] == "(0.1)**2"
    # initval captured
    assert parsed["initval"].get("y") == "0"


def test_read_mod_integration():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "toy.mod"
        p.write_text(SIMPLE_MOD, encoding="utf-8")

        m = read_mod(p)
        # Parameters and shocks wired
        assert "beta" in m.parameters
        assert "rho" in m.parameters
        assert "sigma" in m.parameters
        assert any(str(s) == "e" for s in m.shocks)
        assert any(str(v) == "y" for v in m.variables)

        lin = m.compile_model()
        p0 = m.p0()  # [beta, rho, sigma]
        # smoke test likelihood
        ll = lin.log_lik(p0)
        assert np.isfinite(ll)


ADVANCED_MOD = """
// Advanced blocks: varobs, correlations, endval, varexo_det
parameters beta;
beta = 0.99;

var y;
varobs y;
varexo e, u;

model(linear);
  y = e + u;
end;

shocks; stderr e = 0.1; stderr u = 0.2; corr e, u = 0.5; end;

initval; y = 0; end;
endval; y = 0; end;
"""


def test_parse_mod_with_varobs_and_corr():
    parsed = parse_mod_text(ADVANCED_MOD)
    assert set(parsed["shocks"]) == {"e", "u"}
    assert parsed["covariance"]["e"] == "(0.1)**2"
    assert parsed["covariance"]["u"] == "(0.2)**2"
    # correlations captured
    assert ("e", "u", "0.5") in parsed["correlations"]
    # observables captured
    assert parsed["observables"] == ["y"]


def test_translate_covariance_with_corr():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "adv.mod"
        p.write_text(ADVANCED_MOD, encoding="utf-8")
        m = read_mod(p)
        # prepare matrices so QQ is available
        m.python_sims_matrices()
        p0 = m.p0()
        # evaluate covariance numeric
        QQ = m.QQ(p0)
        # expected diag: 0.01 and 0.04; off-diag: 0.5*0.1*0.2 = 0.01
        assert_allclose(QQ[0, 0], 0.01, rtol=1e-12, atol=1e-12)
        assert_allclose(QQ[1, 1], 0.04, rtol=1e-12, atol=1e-12)
        assert_allclose(QQ[0, 1], 0.01, rtol=1e-12, atol=1e-12)
        assert_allclose(QQ[1, 0], 0.01, rtol=1e-12, atol=1e-12)
