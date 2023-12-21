import unittest
import os
from dsge.parse_yaml import navigate_path, yaml, read_yaml

class TestIncludeConstructor(unittest.TestCase):
     
    def setUp(self):
        self.data = {
            "file1": {
                "key1": "val1",
                "key2": "val2"
            },
            "file2": {
                "keyA": "valA"
            }
        }
        self.path1 = ["file1", "key1"]
        self.path2 = ["file2", "keyA"]
        self.path3 = ["file3", "key1"]

    def test_navigate_path(self):
        self.assertEqual(navigate_path(self.data, self.path1), "val1")
        self.assertEqual(navigate_path(self.data, self.path2), "valA")
        self.assertEqual(navigate_path(self.data, self.path3), None)

    def test_include(self):
        with open('dsge/tests/yaml/check_include.yaml','r') as f:
            re = yaml.safe_load(f)
        self.assertEqual(re, {'key': 'value', 'key2' : 'value2'})

    def test_schema(self):
        fhp  = read_yaml('dsge/examples/fhp/partial_equilibrium.yaml')
        dsge  = read_yaml('dsge/examples/sw/sw.yaml')

if __name__ == "__main__":
    unittest.main()
