import numpy as np
from numpy.testing import assert_allclose

from dsge.endogenous_horizon_switching import (
    EndogenousHorizonSwitchingModel,
    LinearMarginalCostSchedule,
    choose_k_star,
)


def _A(beta: float, rho_y: float, j: int) -> float:
    br = beta * rho_y
    return float((1.0 - br ** (j + 1)) / (1.0 - br))


def _b_j(kappa: float, xi: float, beta: float, rho_y: float, j: int) -> float:
    return float(kappa * xi * _A(beta, rho_y, j))


def _q_j(beta: float, j: int) -> float:
    return float(beta ** (j + 1))


def _pi_hat(y_t: float, pi_bar_t: float, *, b: float, q: float) -> float:
    return float(b * y_t + q * pi_bar_t)


def _p_hat(theta: float, pi_hat_t: float) -> float:
    return float(theta / (1.0 - theta) * pi_hat_t)


def _delta_p_analytic(
    *,
    beta: float,
    theta: float,
    kappa: float,
    xi: float,
    rho_y: float,
    y_t: float,
    pi_bar_t: float,
    j_plus_1: int,
) -> float:
    j_plus_1 = int(j_plus_1)
    return float(
        (beta ** j_plus_1)
        * (theta / (1.0 - theta))
        * (kappa * xi * (rho_y ** j_plus_1) * y_t - (1.0 - beta) * pi_bar_t)
    )


def _mb_analytic(
    *,
    D_pp: float,
    beta: float,
    theta: float,
    delta_p: float,
) -> float:
    # MB = (-D_pp)/(2(1-δ)) * (Δp)^2, δ=βθ.
    if not (D_pp < 0):
        raise ValueError("D_pp must be negative so that -D_pp>0.")
    delta = beta * theta
    return float((-D_pp) / (2.0 * (1.0 - delta)) * (delta_p**2))


def _choose_j_analytic(
    *,
    y_t: float,
    pi_bar_t: float,
    Jmax: int,
    a: float,
    b: float,
    params: dict,
) -> int:
    cost = LinearMarginalCostSchedule(a, b)
    for j in range(0, int(Jmax)):
        dp = _delta_p_analytic(
            beta=params["beta"],
            theta=params["theta"],
            kappa=params["kappa"],
            xi=params["xi"],
            rho_y=params["rho_y"],
            y_t=y_t,
            pi_bar_t=pi_bar_t,
            j_plus_1=j + 1,
        )
        mb = _mb_analytic(D_pp=params["D_pp"], beta=params["beta"], theta=params["theta"], delta_p=dp)
        if mb < cost.delta_tau(j + 1):
            return j
    return int(Jmax)


def _simulate_analytic(
    *,
    T: int,
    seed: int,
    Jmax: int,
    a: float,
    b: float,
    params: dict,
    y0: float = 0.0,
    pi_bar0: float = 0.0,
):
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 1,))
    pi_bar = np.zeros((T + 1,))
    pi = np.zeros((T,))
    j_star = np.zeros((T,), dtype=int)
    eps = rng.standard_normal(size=(T,))

    y[0] = float(y0)
    pi_bar[0] = float(pi_bar0)

    for t in range(T):
        j = _choose_j_analytic(y_t=y[t], pi_bar_t=pi_bar[t], Jmax=Jmax, a=a, b=b, params=params)
        j_star[t] = j

        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)
        pi[t] = _pi_hat(y[t], pi_bar[t], b=b_j, q=q_j)

        pi_bar[t + 1] = (1.0 - params["gamma"]) * pi_bar[t] + params["gamma"] * pi[t]
        y[t + 1] = params["rho_y"] * y[t] + params["sigma_y"] * eps[t]

    return {"y": y, "pi_bar": pi_bar, "pi": pi, "j": j_star, "eps": eps}


def _pe_engine(*, Jmax: int, a: float, b: float, params: dict) -> EndogenousHorizonSwitchingModel:
    # solve_given_regime for scalar horizon j (one component).
    def solve_given_regime(para: np.ndarray, s):
        j = int(s[0])
        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)

        TT = np.array(
            [
                [params["rho_y"], 0.0],
                [params["gamma"] * b_j, 1.0 - params["gamma"] + params["gamma"] * q_j],
            ],
            dtype=float,
        )
        RR = np.array([[params["sigma_y"]], [0.0]], dtype=float)
        ZZ = np.array([[b_j, q_j]], dtype=float)
        DD = np.zeros((1,), dtype=float)
        QQ = np.array([[1.0]], dtype=float)
        HH = np.array([[0.0]], dtype=float)
        return TT, RR, ZZ, DD, QQ, HH

    def info_func(x_t: np.ndarray, t: int, chosen):
        return {"y": float(x_t[0]), "pi_bar": float(x_t[1])}

    def policy_object(para: np.ndarray, info_t, component: str, k: int, chosen):
        j = int(k)
        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)
        pi_j = _pi_hat(info_t["y"], info_t["pi_bar"], b=b_j, q=q_j)
        return _p_hat(params["theta"], pi_j)

    lam = (-params["D_pp"]) / (1.0 - params["beta"] * params["theta"])

    model = EndogenousHorizonSwitchingModel(
        components=["pricing"],
        k_max={"pricing": Jmax},
        cost_params={"pricing": (a, b)},
        lam={"pricing": lam},
        solve_given_regime=solve_given_regime,
        policy_object=policy_object,
        info_func=info_func,
    )
    return model


def _simulate_engine_with_given_shocks(
    *,
    model: EndogenousHorizonSwitchingModel,
    T: int,
    eps: np.ndarray,
    params_vec,
    x0: np.ndarray | None = None,
):
    x = np.zeros((2,), dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
    pi = np.zeros((T,), dtype=float)
    j_star = np.zeros((T,), dtype=int)

    for t in range(T):
        s_t = model.choose_regime(params_vec, x, t=t)
        j_star[t] = int(s_t[0])
        TT, RR, ZZ, DD, QQ, HH = model.get_mats(params_vec, s_t)
        pi[t] = float((ZZ @ x + DD)[0])
        x = TT @ x + (RR[:, 0] * eps[t])

    return {"pi": pi, "j": j_star}


def test_pe_exact_regime_selection_and_inflation_match():
    params = {
        "rho_y": 0.85,
        "sigma_y": 0.25,
        "beta": 0.99,
        "kappa": 0.05,
        "xi": 1.0,
        "theta": 0.75,
        "gamma": 0.3,
        "D_pp": -2.0,
    }
    T = 40
    seed = 123
    Jmax = 8
    a, b = 1e-4, 0.0

    ana = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a, b=b, params=params)
    model = _pe_engine(Jmax=Jmax, a=a, b=b, params=params)
    out = _simulate_engine_with_given_shocks(
        model=model, T=T, eps=ana["eps"], params_vec=np.zeros((1,))
    )

    assert np.array_equal(out["j"], ana["j"])
    assert_allclose(out["pi"], ana["pi"], rtol=0, atol=1e-12)


def test_pe_fixed_regime_collapse_j0_and_jmax():
    params = {
        "rho_y": 0.85,
        "sigma_y": 0.25,
        "beta": 0.99,
        "kappa": 0.05,
        "xi": 1.0,
        "theta": 0.75,
        "gamma": 0.3,
        "D_pp": -2.0,
    }
    T = 30
    seed = 7
    Jmax = 6

    # Force j*=0 by very high marginal costs.
    a0, b0 = 1e9, 0.0
    ana0 = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a0, b=b0, params=params)
    model0 = _pe_engine(Jmax=Jmax, a=a0, b=b0, params=params)
    out0 = _simulate_engine_with_given_shocks(model=model0, T=T, eps=ana0["eps"], params_vec=np.zeros((1,)))
    assert np.all(out0["j"] == 0)
    assert_allclose(out0["pi"], ana0["pi"], rtol=0, atol=1e-12)

    # Force j*=Jmax by very low marginal costs.
    a1, b1 = 1e-20, 0.0
    x0 = np.array([1.0, 0.2])
    ana1 = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a1, b=b1, params=params, y0=x0[0], pi_bar0=x0[1])
    model1 = _pe_engine(Jmax=Jmax, a=a1, b=b1, params=params)
    out1 = _simulate_engine_with_given_shocks(model=model1, T=T, eps=ana1["eps"], params_vec=np.zeros((1,)), x0=x0)
    assert np.all(out1["j"] == Jmax)
    assert_allclose(out1["pi"], ana1["pi"], rtol=0, atol=1e-12)


def test_pe_stopping_rule_invariants_hold():
    params = {
        "rho_y": 0.85,
        "sigma_y": 0.25,
        "beta": 0.99,
        "kappa": 0.05,
        "xi": 1.0,
        "theta": 0.75,
        "gamma": 0.3,
        "D_pp": -2.0,
    }
    Jmax = 5
    a, b = 1e-4, 1e-5
    model = _pe_engine(Jmax=Jmax, a=a, b=b, params=params)

    x = np.array([0.5, -0.2])
    s = model.choose_regime(np.zeros((1,)), x, t=0)
    j_star = int(s[0])

    # Check stopping rule consistency at this state:
    info = {"y": float(x[0]), "pi_bar": float(x[1])}
    chosen = {}
    cost = LinearMarginalCostSchedule(a, b)
    lam = (-params["D_pp"]) / (1.0 - params["beta"] * params["theta"])

    def policy(params_vec, info_t, component, k, chosen_regime):
        j = int(k)
        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)
        pi_j = _pi_hat(info_t["y"], info_t["pi_bar"], b=b_j, q=q_j)
        return _p_hat(params["theta"], pi_j)

    def mb(params_vec, info_t, component, k_plus_1, chosen_regime):
        p1 = policy(params_vec, info_t, component, k_plus_1, chosen_regime)
        p0 = policy(params_vec, info_t, component, k_plus_1 - 1, chosen_regime)
        return 0.5 * lam * (p1 - p0) ** 2

    # Earlier steps: MB >= cost, stop step: MB < cost (unless capped).
    for j in range(0, j_star):
        assert mb(np.zeros((1,)), info, "pricing", j + 1, chosen) >= cost.delta_tau(j + 1)
    if j_star < Jmax:
        assert mb(np.zeros((1,)), info, "pricing", j_star + 1, chosen) < cost.delta_tau(j_star + 1)


def test_pe_time_varying_arma_reproduces_demand_shocks():
    # Conditional reduced-form oracle: given realized regimes, compute the implied ARMA(2,1)
    # innovations for inflation and verify they match the underlying demand shocks.
    params = {
        "rho_y": 0.85,
        "sigma_y": 0.25,
        "beta": 0.99,
        "kappa": 0.05,
        "xi": 1.0,
        "theta": 0.75,
        "gamma": 0.3,
        "D_pp": -2.0,
    }
    T = 60
    seed = 999
    Jmax = 8
    a, b = 1e-4, 0.0

    ana = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a, b=b, params=params)
    model = _pe_engine(Jmax=Jmax, a=a, b=b, params=params)
    out = _simulate_engine_with_given_shocks(model=model, T=T, eps=ana["eps"], params_vec=np.zeros((1,)))

    pi = out["pi"]
    j = out["j"]
    eps = ana["eps"]

    # Build per-t matrices for the conditional linear system.
    # State s_t = [y_t, pi_bar_t], shock is eps_t.
    def mats_for_j(jt: int):
        b_jt = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], int(jt))
        q_jt = _q_j(params["beta"], int(jt))
        Tm = np.array(
            [
                [params["rho_y"], 0.0],
                [params["gamma"] * b_jt, 1.0 - params["gamma"] + params["gamma"] * q_jt],
            ]
        )
        Zm = np.array([[b_jt, q_jt]])
        Rm = np.array([[params["sigma_y"]], [0.0]])
        return Tm, Zm, Rm

    # ARMA(2,1) derivation (time-varying):
    # π_t = φ1_t π_{t-1} + φ2_t π_{t-2} + ψ_t eps_{t-1} + (Z_t R) eps_t
    # where [φ2_t, φ1_t] = Z_t T_{t-1} T_{t-2} M^{-1} with
    #   M = [[Z_{t-2}], [Z_{t-1} T_{t-2}]].
    eps_hat = np.zeros_like(eps)
    for t in range(2, T):
        T_t2, Z_t2, R_t2 = mats_for_j(j[t - 2])
        T_t1, Z_t1, R_t1 = mats_for_j(j[t - 1])
        T_t, Z_t, R_t = mats_for_j(j[t])

        M = np.vstack([Z_t2, Z_t1 @ T_t2])
        Minv = np.linalg.inv(M)
        g = (Z_t @ T_t1 @ T_t2) @ Minv  # 1x2
        phi2, phi1 = float(g[0, 0]), float(g[0, 1])

        psi = float((Z_t @ T_t1 @ R_t1)[0, 0] - phi1 * (Z_t1 @ R_t1)[0, 0])
        ztr = float((Z_t @ R_t)[0, 0])

        # Note indexing: eps[t] is used to propagate y_{t+1}; thus π_t loads on eps[t-1].
        w_t = pi[t] - phi1 * pi[t - 1] - phi2 * pi[t - 2] - psi * eps[t - 2]
        eps_hat[t] = w_t / ztr

    assert_allclose(eps_hat[2:], eps[1:-1], rtol=0, atol=1e-10)


def test_pe_particle_filter_smoke():
    params = {
        "rho_y": 0.85,
        "sigma_y": 0.25,
        "beta": 0.99,
        "kappa": 0.05,
        "xi": 1.0,
        "theta": 0.75,
        "gamma": 0.3,
        "D_pp": -2.0,
    }
    T = 25
    seed = 2025
    Jmax = 6
    a, b = 1e-4, 0.0
    sigma_eta = 0.05

    ana = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a, b=b, params=params)

    # Engine with measurement error for the particle filter.
    def solve_given_regime(para: np.ndarray, s):
        j = int(s[0])
        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)

        TT = np.array(
            [
                [params["rho_y"], 0.0],
                [params["gamma"] * b_j, 1.0 - params["gamma"] + params["gamma"] * q_j],
            ],
            dtype=float,
        )
        RR = np.array([[params["sigma_y"]], [0.0]], dtype=float)
        ZZ = np.array([[b_j, q_j]], dtype=float)
        DD = np.zeros((1,), dtype=float)
        QQ = np.array([[1.0]], dtype=float)
        HH = np.array([[sigma_eta**2]], dtype=float)
        return TT, RR, ZZ, DD, QQ, HH

    def info_func(x_t: np.ndarray, t: int, chosen):
        return {"y": float(x_t[0]), "pi_bar": float(x_t[1])}

    def policy_object(para: np.ndarray, info_t, component: str, k: int, chosen):
        j = int(k)
        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)
        pi_j = _pi_hat(info_t["y"], info_t["pi_bar"], b=b_j, q=q_j)
        return _p_hat(params["theta"], pi_j)

    lam = (-params["D_pp"]) / (1.0 - params["beta"] * params["theta"])
    model = EndogenousHorizonSwitchingModel(
        components=["pricing"],
        k_max={"pricing": Jmax},
        cost_params={"pricing": (a, b)},
        lam={"pricing": lam},
        solve_given_regime=solve_given_regime,
        policy_object=policy_object,
        info_func=info_func,
    )

    rng = np.random.default_rng(seed)
    y_obs = ana["pi"] + sigma_eta * rng.standard_normal(size=(T,))

    ll, stats = model.pf_loglik(np.zeros((1,)), y_obs, nparticles=800, seed=0)
    assert np.isfinite(ll)
    assert stats["k_mean"].shape == (T, 1)
    assert np.all(stats["k_mean"] >= 0.0)
    assert np.all(stats["k_mean"] <= float(Jmax))
