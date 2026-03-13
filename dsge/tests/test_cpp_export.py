import tempfile
import unittest

from scipy.stats import invgamma, lognorm

from dsge import read_yaml
from dsge.resource_utils import resource_path
from dsge.translate import translate
from dsge.translate_cpp import generate_dsge_logprior, generate_dsge_prior_draws

class TestCppExport(unittest.TestCase):
    def test_cpp_export_for_fhp_model(self):
        # Use the in-repo FHP example
        with resource_path('examples/fhp/fhp.yaml') as p:
            model = read_yaml(str(p))

        with tempfile.TemporaryDirectory(prefix="dsge_cpp_export_") as td:
            translate(model, output_dir=td, language="cpp")

    def test_cpp_prior_codegen_supports_invgamma_alias(self):
        prior = [invgamma(3.0, scale=0.25)]

        logprior = generate_dsge_logprior(prior)
        draws = generate_dsge_prior_draws(prior)

        self.assertIn("inv_gamma_lpdf", logprior)
        self.assertIn("inv_gamma_rng", draws)

    def test_cpp_prior_codegen_rejects_unknown_distribution(self):
        prior = [lognorm(s=0.2, scale=1.0)]

        with self.assertRaises(NotImplementedError):
            generate_dsge_logprior(prior)

        with self.assertRaises(NotImplementedError):
            generate_dsge_prior_draws(prior)

if __name__ == "__main__":
    unittest.main()
