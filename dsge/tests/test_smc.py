import numpy as np


class DummyModel:
    def __init__(self):
        from scipy.stats import norm

        from dsge.Prior import Prior

        self.parameter_names = ["theta1", "theta2"]
        self.prior = Prior([norm(loc=0.0, scale=1.0), norm(loc=0.0, scale=1.0)])

    def log_lik(self, para, use_cache=False, **kwargs):
        para = np.asarray(para, dtype=float)
        mu = np.array([1.0, -1.0])
        sig = 0.5
        return float(-0.5 * np.sum(((para - mu) / sig) ** 2))


class PidModel:
    parameter_names = ["theta1", "theta2"]
    prior = None

    def log_lik(self, para, use_cache=False, **kwargs):
        import os

        return float(os.getpid())


def test_smc_dummy_model_runs():
    from dsge.smc import SMCOptions, smc_estimate

    opts = SMCOptions(npart=200, nphi=15, bend=2.0, seed=0, nintmh=1, nblocks=1, npriorextra=0)
    res = smc_estimate(DummyModel(), options=opts, n_workers=1, parallel="none")

    assert res.particles.shape == (opts.npart, 2)
    assert res.weights.shape == (opts.npart,)
    assert np.isclose(res.weights.sum(), 1.0)
    assert res.phi_schedule[0] == 0.0
    assert res.phi_schedule[-1] == 1.0
    assert len(res.stage_stats) == opts.nphi - 1

    # Threaded likelihood evaluation path (should also run).
    res_thr = smc_estimate(DummyModel(), options=opts, n_workers=2, parallel="thread")
    assert res_thr.particles.shape == (opts.npart, 2)
    assert np.isclose(res_thr.weights.sum(), 1.0)

    # Multi-process likelihood evaluation path (should also run).
    res_proc = smc_estimate(DummyModel(), options=opts, n_workers=2, parallel="process")
    assert res_proc.particles.shape == (opts.npart, 2)
    assert np.isclose(res_proc.weights.sum(), 1.0)


def test_process_batch_evaluator_uses_multiple_workers():
    from dsge.smc import _LogLikBatchEvaluator

    particles = np.zeros((40, 2))
    with _LogLikBatchEvaluator(PidModel(), log_lik_kwargs={}, n_workers=2, parallel="process") as ev:
        out = ev.eval_many(particles)

    assert len({int(x) for x in out}) >= 2


def test_smc_smoke_nkmp_model():
    from dsge import read_yaml

    m = read_yaml("dsge/examples/nkmp/nkmp.yaml")
    res = m.estimate_smc(
        order=1,
        backend="local",
        smc_options={
            "npart": 40,
            "nphi": 5,
            "bend": 2.0,
            "seed": 123,
            "nintmh": 1,
            "nblocks": 1,
            "initial_scale": 0.10,
            "npriorextra": 50,
            "verbose": False,
        },
        n_workers=1,
        parallel="none",
    )
    assert res.particles.shape[0] == 40
    assert res.particles.shape[1] == len(res.parameter_names)
    assert np.isfinite(res.log_lik).all()


def test_smc_smoke_nkmp_model_allows_state_space_reduction():
    from dsge import read_yaml

    m = read_yaml("dsge/examples/nkmp/nkmp.yaml")
    res = m.estimate_smc(
        order=1,
        backend="local",
        smc_options={
            "npart": 10,
            "nphi": 3,
            "bend": 2.0,
            "seed": 123,
            "nintmh": 1,
            "nblocks": 1,
            "initial_scale": 0.10,
            "npriorextra": 50,
            "verbose": False,
        },
        log_lik_kwargs={"reduce_state_space": "minimal", "reduce_tol": 1e-10},
        n_workers=1,
        parallel="none",
    )
    assert res.particles.shape[0] == 10
    assert np.isfinite(res.log_lik).all()
