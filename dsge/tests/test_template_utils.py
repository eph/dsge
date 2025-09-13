import pytest
from dsge.template_utils import render_template, build_fhp_placeholders


def test_render_template_basic():
    template = "A={A} B={B}"
    rendered = render_template(template, {'{A}': 1, '{B}': 'two'})
    assert rendered == "A=1 B=two"


def test_render_template_strict_unmatched():
    template = "C={C} D={D}"
    with pytest.raises(ValueError):
        render_template(template, {'{C}': 3})


def test_build_fhp_placeholders():
    ph = build_fhp_placeholders(
        nobs=2, T=3, nvar=4, nval=5, nshock=6, npara=7, neps=8, k=9, t0=0, system='SYS')
    assert ph['{cmodel.yy.shape[1]}'] == '2'
    assert ph['{cmodel.yy.shape[0]}'] == '3'
    assert ph["{len(model['variables'])}"] == '4'
    assert ph['{k}'] == '9'
    assert ph['{system}'] == 'SYS'

