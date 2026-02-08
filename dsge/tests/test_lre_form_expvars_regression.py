import pytest

from dsge import read_yaml


def test_expvars_solves_where_legacy_has_coincident_zeros():
    """
    Regression: the old LRE augmentation (lre_form='legacy') can flag GENSYS
    coincident zeros on large, sparse models. The expvars formulation should
    avoid singular pencils and solve.
    """
    m = read_yaml("dsge/examples/linver/linver_mini_legacy_fail.yaml")
    p0 = m.p0()

    lin_bad = m.compile_model(order=1, lre_form="legacy")
    _tt, _rr, rc_bad = lin_bad.solve_LRE(p0, return_diagnostics=True, scale_equations=True)
    assert rc_bad == 0
    assert lin_bad.last_gensys_diagnostics["coincident_zeros"] is True

    lin_ok = m.compile_model(order=1, lre_form="expvars")
    _tt, _rr, rc_ok = lin_ok.solve_LRE(p0, return_diagnostics=True, scale_equations=True)
    assert rc_ok == 1
    assert lin_ok.last_gensys_diagnostics["coincident_zeros"] is False

