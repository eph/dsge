import re
from typing import Dict, Any


def render_template(template_text: str, placeholders: Dict[str, Any], strict: bool = True) -> str:
    """Render a template by replacing exact placeholder keys with values.

    - Placeholders must be exact substrings (including braces) present in the template.
    - Values are converted to strings.
    - If strict=True, raises ValueError if any placeholders remain in the form {something}.
    """
    rendered = template_text
    for key, val in placeholders.items():
        rendered = rendered.replace(key, str(val))

    if strict:
        # Detect any unreplaced placeholders like {foo}
        if re.search(r"\{[^}]+\}", rendered):
            missing = re.findall(r"\{[^}]+\}", rendered)
            raise ValueError(f"Unresolved placeholders remain in template: {sorted(set(missing))}")

    return rendered


def build_fhp_placeholders(
    *,
    nobs: int,
    T: int,
    nvar: int,
    nval: int,
    nshock: int,
    npara: int,
    neps: int,
    k: int,
    t0: int,
    system: str,
    data: str = "",
    custom_prior_code: str = "",
    p0: str = "",
) -> Dict[str, Any]:
    """Build the known placeholders used by templates/fhp.f90."""
    return {
        '{cmodel.yy.shape[1]}': str(nobs),
        '{cmodel.yy.shape[0]}': str(T),
        "{len(model['variables'])}": str(nvar),
        "{len(model['values'])}": str(nval),
        "{len(model['shocks'])}": str(nshock),
        "{len(model['parameters'])}": str(npara),
        "{len(model['innovations'])}": str(neps),
        '{k}': str(k),
        '{p0}': p0,
        '{t0}': str(t0),
        '{system}': system,
        '{data}': data,
        '{custom_prior_code}': custom_prior_code,
    }

