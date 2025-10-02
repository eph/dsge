from __future__ import annotations

import numpy as np

from unittest import TestCase

from dsge import read_yaml
from dsge.resource_utils import resource_path


class TestABCDRepresentation(TestCase):

    def test_shapes_and_identity(self):
        """ABCD returns correct shapes and implements Z(T s + R e)."""
        with resource_path('examples/pi/pi.yaml') as p:
            model = read_yaml(str(p))

        m = model.compile_model()
        para = [1.0, 1.2]  # [sigw, R]

        # Pull both system matrices and ABCD
        CC, TT, RR, QQ, DD, ZZ, HH = m.system_matrices(para)
        A, B, C, D = m.abcd_representation(para)

        # Shape checks
        ns = TT.shape[0]
        neps = RR.shape[1]
        ny = ZZ.shape[0]

        self.assertEqual(A.shape, TT.shape)
        self.assertEqual(B.shape, RR.shape)
        self.assertEqual(C.shape, (ny, ns))
        self.assertEqual(D.shape, (ny, neps))

        # Direct equality checks for definitions
        np.testing.assert_allclose(A, TT)
        np.testing.assert_allclose(B, RR)
        np.testing.assert_allclose(C, ZZ @ TT)
        np.testing.assert_allclose(D, ZZ @ RR)

        # Behavioral identity: y_t = C s_{t-1} + D e_t equals Z(T s_{t-1} + R e_t)
        rng = np.random.default_rng(123)
        s_prev = rng.normal(size=ns)
        e_t = rng.normal(size=neps)

        y1_from_abcd = (C @ s_prev) + (D @ e_t)
        y1_direct = ZZ @ (TT @ s_prev + RR @ e_t)
        np.testing.assert_allclose(y1_from_abcd, y1_direct)

