from dsge import read_yaml


def test_legacy_loop_backend_handles_lagged_expectation_placeholders():
    """
    Regression: the loop differentiation backend must handle `llist` symbols that are
    not `Variable` instances (e.g. legacy `__LAGGED_*` placeholders for fvars).
    """
    m = read_yaml("dsge/examples/lre_form/tiny_forward_expectation.yaml")
    p0 = m.p0()

    # Force the loop backend to mirror large-model behavior.
    m.python_sims_matrices(method="loop", lre_form="legacy")
    lin = m.compile_model(order=1, lre_form="legacy")
    _tt, _rr, rc = lin.solve_LRE(p0, return_diagnostics=True, scale_equations=True)
    assert rc == 1
    assert lin.last_gensys_diagnostics["coincident_zeros"] is False

    # expvars should also solve (and does not rely on placeholders).
    m2 = read_yaml("dsge/examples/lre_form/tiny_forward_expectation.yaml")
    m2.python_sims_matrices(method="loop", lre_form="expvars")
    lin2 = m2.compile_model(order=1, lre_form="expvars")
    _tt, _rr, rc2 = lin2.solve_LRE(p0, return_diagnostics=True, scale_equations=True)
    assert rc2 == 1
    assert lin2.last_gensys_diagnostics["coincident_zeros"] is False

