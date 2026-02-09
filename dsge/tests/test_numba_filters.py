import numpy as np

from dsge.filters import chand_recursion, chand_recursion_py, kalman_filter, kalman_filter_py


def _stable_transition(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n))
    # Scale so spectral radius < 1.
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a / (1.5 * max(rho, 1e-6))


def test_chand_recursion_numba_matches_python():
    rng = np.random.default_rng(0)
    t = 200
    ns = 15
    ny = 5
    neps = 3

    y = rng.normal(size=(t, ny))
    tt = _stable_transition(rng, ns)
    rr = rng.normal(size=(ns, neps))
    qq = np.eye(neps)
    cc = np.zeros(ns)
    dd = np.zeros(ny)
    zz = rng.normal(size=(ny, ns))
    hh = 0.1 * np.eye(ny)
    a0 = np.zeros(ns)
    p0 = np.eye(ns)

    ll_py = chand_recursion_py(y, cc, tt, rr, qq, dd, zz, hh, a0, p0)
    ll_nb = chand_recursion(y, cc, tt, rr, qq, dd, zz, hh, a0, p0)
    assert np.isfinite(ll_py)
    assert float(abs(ll_py - ll_nb)) == 0.0


def test_kalman_filter_numba_matches_python():
    rng = np.random.default_rng(1)
    t = 200
    ns = 15
    ny = 5
    neps = 3

    y = rng.normal(size=(t, ny))
    tt = _stable_transition(rng, ns)
    rr = rng.normal(size=(ns, neps))
    qq = np.eye(neps)
    cc = np.zeros(ns)
    dd = np.zeros(ny)
    zz = rng.normal(size=(ny, ns))
    hh = 0.1 * np.eye(ny)
    a0 = np.zeros(ns)
    p0 = np.eye(ns)

    ll_py = kalman_filter_py(y, cc, tt, rr, qq, dd, zz, hh, a0, p0)
    ll_nb = kalman_filter(y, cc, tt, rr, qq, dd, zz, hh, a0, p0)
    assert np.isfinite(ll_py)
    assert float(abs(ll_py - ll_nb)) == 0.0

