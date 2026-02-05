from unittest import TestCase
from unittest.mock import patch

from dsge import read_yaml
from dsge.resource_utils import resource_path


class TestPythonSimsMatricesCache(TestCase):
    def test_python_sims_matrices_returns_early_when_compiled(self):
        with resource_path("examples/ar1/ar1.yaml") as p:
            m = read_yaml(str(p))

        # First compile populates the numeric lambdified callables.
        m.python_sims_matrices()

        # If the cache check regresses, this would try to build symbolic matrices again.
        with patch("sympy.matrices.matrixbase.MatrixBase.jacobian", side_effect=AssertionError("should not be called")):
            out = m.python_sims_matrices(method="jacobian")
        self.assertIsNone(out)

