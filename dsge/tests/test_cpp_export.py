import unittest
from dsge import read_yaml
from dsge.resource_utils import resource_path
from dsge.translate import translate

class TestCppExport(unittest.TestCase):
    def test_cpp_export_for_fhp_model(self):
        # Use the in-repo FHP example
        with resource_path('examples/fhp/fhp.yaml') as p:
            model = read_yaml(str(p))

        # Calling translate with language "cpp" should trigger NotImplementedError.
        #with self.assertRaises(NotImplementedError) as context:
        translate(model, output_dir="_tmp_cpp_export", language="cpp")
        #self.assertEqual(str(context.exception), "C++ export not implemented.") 

if __name__ == "__main__":
    unittest.main()
