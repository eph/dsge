import numpy as np

from unittest import TestCase

from dsge import read_yaml
from dsge.resource_utils import resource_path


class TestDeterminacyReport(TestCase):
    def test_report_smoke_and_cache(self):
        with resource_path("examples/nkmp/dsge1.yaml") as p:
            m = read_yaml(str(p))

        compiled = m.compile_model()

        p0 = np.array(
            [
                1.62398783,
                0.47671893,
                1.51729311,
                0.4416236,
                0.43724069,
                0.65953619,
                0.60440589,
                0.51634896,
                5.82589805,
                0.68667388,
                0.54936489,
                0.53220526,
                3.20302156,
            ]
        )

        # Should work even when cache is empty.
        compiled.solve_LRE(p0, use_cache=True)

        rep = compiled.determinacy_report(p0, qz_criteria=[1 + 1e-6, 1 + 1e-8], use_cache=True)
        self.assertIn("by_qz", rep)
        self.assertEqual(len(rep["by_qz"]), 2)
        self.assertTrue(all(r["rc"] == 1 for r in rep["by_qz"]))

