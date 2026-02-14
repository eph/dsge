import numpy as np
from numpy.testing import assert_allclose

from importlib.resources import files

from dsge.parse_yaml import read_yaml


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


def _mb_analytic(*, D_pp: float, beta: float, theta: float, delta_p: float) -> float:
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
    params: dict,
) -> int:
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
        if mb < float(a):
            return j
    return int(Jmax)


def _simulate_analytic(
    *,
    T: int,
    seed: int,
    Jmax: int,
    a: float,
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
        j = _choose_j_analytic(y_t=y[t], pi_bar_t=pi_bar[t], Jmax=Jmax, a=a, params=params)
        j_star[t] = j

        b_j = _b_j(params["kappa"], params["xi"], params["beta"], params["rho_y"], j)
        q_j = _q_j(params["beta"], j)
        pi[t] = _pi_hat(y[t], pi_bar[t], b=b_j, q=q_j)

        pi_bar[t + 1] = (1.0 - params["gamma"]) * pi_bar[t] + params["gamma"] * pi[t]
        y[t + 1] = params["rho_y"] * y[t] + params["sigma_y"] * eps[t]

    return {"y": y, "pi_bar": pi_bar, "pi": pi, "j": j_star, "eps": eps}


def _simulate_engine_with_given_shocks(*, model, T: int, eps: np.ndarray, params_vec, x0: np.ndarray | None = None):
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


def test_switching_ssm_yaml_pe_matches_analytic():
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
    a = 1e-4

    ana = _simulate_analytic(T=T, seed=seed, Jmax=Jmax, a=a, params=params)

    yaml_path = files("dsge") / "examples" / "switching" / "pe_pricing_switching.yaml"
    model = read_yaml(str(yaml_path))

    out = _simulate_engine_with_given_shocks(model=model, T=T, eps=ana["eps"], params_vec=model.p0)

    assert np.array_equal(out["j"], ana["j"])
    assert_allclose(out["pi"], ana["pi"], rtol=0, atol=1e-12)
