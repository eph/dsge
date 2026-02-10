import numpy as np


def _toy_system():
    # 3-state system where the 3rd state is unobservable and does not feed into the first two.
    TT = np.array(
        [
            [0.90, 0.10, 0.00],
            [0.00, 0.80, 0.00],
            [0.00, 0.00, 0.70],
        ],
        dtype=float,
    )
    RR = np.array([[1.0], [0.5], [0.3]], dtype=float)
    QQ = np.array([[1.0]], dtype=float)
    CC = np.zeros(3, dtype=float)

    ZZ = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    DD = np.zeros(2, dtype=float)
    HH = 0.01 * np.eye(2, dtype=float)
    return CC, TT, RR, QQ, DD, ZZ, HH


def test_reduce_state_space_drops_unobservable_state():
    from dsge.state_space_reduction import reduce_state_space

    CC, TT, RR, QQ, DD, ZZ, HH = _toy_system()
    CC2, TT2, RR2, QQ2, DD2, ZZ2, HH2, A02, P02, info = reduce_state_space(
        CC,
        TT,
        RR,
        QQ,
        DD,
        ZZ,
        HH,
        A0=np.zeros_like(CC),
        P0=None,
        mode="observable",
        tol=1e-12,
    )

    assert info.ns_original == 3
    assert info.ns_final == 2
    assert TT2.shape == (2, 2)
    assert RR2.shape[0] == 2
    assert ZZ2.shape == (2, 2)
    assert A02 is not None and A02.shape == (2,)
    assert P02 is None
    assert np.allclose(QQ2, QQ)
    assert np.allclose(DD2, DD)
    assert np.allclose(HH2, HH)


def test_reduce_state_space_can_drop_uncontrollable_states_when_safe():
    from dsge.state_space_reduction import reduce_state_space

    # Second state is not shocked; with CC=A0=0 and P0 supported on state 1 only,
    # it is safe to drop the uncontrollable direction.
    TT = np.array([[0.9, 0.0], [0.0, 0.8]], dtype=float)
    RR = np.array([[1.0], [0.0]], dtype=float)
    QQ = np.array([[1.0]], dtype=float)
    CC = np.zeros(2, dtype=float)

    # Observe only the uncontrollable state (it stays identically zero).
    ZZ = np.array([[0.0, 1.0]], dtype=float)
    DD = np.zeros(1, dtype=float)
    HH = 0.01 * np.eye(1, dtype=float)

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    CC2, TT2, RR2, QQ2, DD2, ZZ2, HH2, A02, P02, info = reduce_state_space(
        CC,
        TT,
        RR,
        QQ,
        DD,
        ZZ,
        HH,
        A0=np.zeros_like(CC),
        P0=P0,
        mode="minimal",
        tol=1e-12,
    )

    assert info.dropped_controllable is True
    assert info.ns_final == 0
    assert TT2.shape == (0, 0)
    assert RR2.shape == (0, 1)
    assert ZZ2.shape == (1, 0)
    assert P02 is not None and P02.shape == (0, 0)
    assert np.allclose(QQ2, QQ)
    assert np.allclose(DD2, DD)
    assert np.allclose(HH2, HH)


def test_log_lik_reduction_matches_full():
    import pandas as p

    from dsge.StateSpaceModel import StateSpaceModel

    CC, TT, RR, QQ, DD, ZZ, HH = _toy_system()

    rng = np.random.default_rng(0)
    y = p.DataFrame(rng.normal(size=(60, ZZ.shape[0])))

    m = StateSpaceModel(
        y,
        CC=lambda *_args, **_kwargs: CC,
        TT=lambda *_args, **_kwargs: TT,
        RR=lambda *_args, **_kwargs: RR,
        QQ=lambda *_args, **_kwargs: QQ,
        DD=lambda *_args, **_kwargs: DD,
        ZZ=lambda *_args, **_kwargs: ZZ,
        HH=lambda *_args, **_kwargs: HH,
        t0=0,
    )

    para = np.zeros((0,), dtype=float)
    ll_full = float(m.log_lik(para))
    ll_red = float(m.log_lik(para, reduce_state_space="observable", reduce_tol=1e-12))
    assert np.isfinite(ll_full)
    assert np.isfinite(ll_red)
    assert float(abs(ll_full - ll_red)) < 1e-8
