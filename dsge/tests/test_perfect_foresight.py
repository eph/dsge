import numpy as np
from numpy.testing import assert_allclose

from dsge import read_yaml
from dsge.resource_utils import resource_path


def test_perfect_foresight_ar1_impulse():
    # Load AR(1) example
    with resource_path('examples/ar1/ar1.yaml') as p:
        m = read_yaml(str(p))
    p0 = m.p0()
    ar1 = m.compile_model()

    # One-period unit shock, then zeros
    T = 6
    eps = np.zeros((T, 1))
    eps[0, 0] = 1.0

    res = ar1.perfect_foresight(p0, eps)
    y = res['observables'].values.squeeze()

    # For AR(1): y_t = rho^t when starting at s0=0 and e0=1
    expected = np.array([0.85**t for t in range(T)])
    assert_allclose(y, expected, atol=1e-10, rtol=1e-10)
