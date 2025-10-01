import numpy as np

from dsge import read_yaml


def test_prior_rvs_returns_parameter_vector_and_system_matrices():
    model = read_yaml('dsge/examples/ar1/ar1.yaml')
    compiled = model.compile_model()

    rng = np.random.default_rng(123)
    draw = compiled.prior.rvs(random_state=rng)

    assert draw.shape == (len(compiled.parameter_names),)

    matrices = compiled.system_matrices(draw)
    for mat in matrices:
        assert np.all(np.isfinite(np.asarray(mat)))

    multiple = compiled.prior.rvs(size=5, random_state=np.random.default_rng(456))
    assert multiple.shape == (5, len(compiled.parameter_names))
