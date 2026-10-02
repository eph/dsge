"""Resource paths must originate from packaged bytes, not a shared cache."""
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from dsge import resource_utils


class ResourceSecurityTests(unittest.TestCase):
    def test_planted_legacy_cache_is_not_used(self):
        with tempfile.TemporaryDirectory() as temp:
            poisoned = Path(temp) / "dsge_resources_cache/examples/ar1/ar1.yaml"
            poisoned.parent.mkdir(parents=True)
            poisoned.write_text("poisoned test fixture")
            with patch("tempfile.gettempdir", return_value=temp):
                result = resource_utils.resource_file_path("examples/ar1/ar1.yaml")
            self.assertNotEqual(result, poisoned)
            self.assertEqual(result.read_bytes(), resource_utils._resource("examples/ar1/ar1.yaml").read_bytes())
            self.assertEqual(poisoned.read_text(), "poisoned test fixture")

    def test_zip_resource_is_private_and_lives_until_stack_close(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            archive = Path(temp) / "resources.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("model.yaml", "trusted: package bytes\n")
            bundle = stack.enter_context(zipfile.ZipFile(archive))
            resource = zipfile.Path(bundle, "model.yaml")
            resource_stack = ExitStack()
            with patch.object(resource_utils, "_resource", return_value=resource), \
                 patch.object(resource_utils, "_RESOURCE_STACK", resource_stack):
                result = resource_utils.resource_file_path("model.yaml")
                second = resource_utils.resource_file_path("model.yaml")
            self.assertEqual(result.read_text(), "trusted: package bytes\n")
            self.assertEqual(result.stat().st_mode & 0o077, 0)
            self.assertNotEqual(result, second)
            resource_stack.close()
            self.assertFalse(result.exists())
            self.assertFalse(second.exists())

    def test_resource_traversal_is_rejected(self):
        for value in ("/etc/passwd", "../pyproject.toml", "examples/../../x", "..\\x", "D:/outside/fixture.yaml", "D:fixture.yaml"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_utils.resource_file_path(value)
