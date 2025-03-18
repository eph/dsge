import os
import unittest
from dsge.FHPRepAgent import FHPRepAgent
from dsge.translate import translate

class TestCppExport(unittest.TestCase):
    def test_cpp_export_for_fhp_model(self):
        # Assuming an FHP model YAML file exists in 'examples/fhp.yaml'
        example_yaml_path = os.path.join(os.path.dirname(__file__), "..", "examples", "fhp.yaml")
        with open(example_yaml_path, "r") as f:
            model_yaml = f.read()
        model = FHPRepAgent.read(model_yaml)

        # Calling translate with language "cpp" should trigger NotImplementedError.
        with self.assertRaises(NotImplementedError) as context:
            translate(model, output_dir="_tmp_cpp_export", language="cpp")
        self.assertEqual(str(context.exception), "C++ export not implemented.")

if __name__ == "__main__":
    unittest.main()
