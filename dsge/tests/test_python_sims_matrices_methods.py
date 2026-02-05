import numpy as np
import sympy
from numpy.testing import assert_allclose

from unittest import TestCase

from dsge import read_yaml
from dsge.resource_utils import resource_path
from dsge.symbols import Parameter


def _eval_sympy_matrix(mat, subs: dict) -> np.ndarray:
    if hasattr(mat, "subs"):
        return np.asarray(mat.subs(subs), dtype=float)
    return np.asarray(mat, dtype=float)


class TestPythonSimsMatricesMethods(TestCase):
    def test_jacobian_and_loop_match(self):
        with resource_path("examples/ar1/ar1.yaml") as p:
            m = read_yaml(str(p))

        p0 = m.p0()
        subs = {Parameter(name): float(val) for name, val in zip(m.parameters, p0)}

        G0_l, G1_l, PSI_l, PPI_l, _, DD_l, ZZ_l, _ = m.python_sims_matrices(matrix_format="symbolic", method="loop")
        G0_j, G1_j, PSI_j, PPI_j, _, DD_j, ZZ_j, _ = m.python_sims_matrices(matrix_format="symbolic", method="jacobian")

        for a, b in [
            (G0_l, G0_j),
            (G1_l, G1_j),
            (PSI_l, PSI_j),
            (PPI_l, PPI_j),
            (DD_l, DD_j),
            (ZZ_l, ZZ_j),
        ]:
            assert_allclose(_eval_sympy_matrix(a, subs), _eval_sympy_matrix(b, subs), rtol=0.0, atol=0.0)

