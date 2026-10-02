"""Includes are a host capability, never a global SafeLoader extension."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dsge.parse_yaml import load_schema, load_yaml, read_yaml


class YAMLIncludeSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "child.yaml").write_text("value: 42\n")

    def test_default_and_unrelated_loader_cannot_include(self):
        text = "!include child.yaml"
        with self.assertRaisesRegex(ValueError, "disabled"):
            load_yaml(text)
        with self.assertRaises(yaml.constructor.ConstructorError):
            yaml.safe_load(text)

    def test_include_is_relative_to_source_and_supports_fragments(self):
        folder = self.root / "nested"
        folder.mkdir()
        (folder / "child.yaml").write_text("value: 73\n")
        self.assertEqual(load_yaml("!include child.yaml#value", include_root=self.root,
                                  source_path=folder / "main.yaml"), 73)

    def test_absolute_parent_and_symlink_escapes_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "fixture.yaml"
            secret.write_text("value: only-a-test-fixture\n")
            (self.root / "link.yaml").symlink_to(secret)
            for path in (str(secret), "../fixture.yaml", "link.yaml", "D:/fixture.yaml", "D:fixture.yaml"):
                with self.subTest(path=path), patch.object(Path, "open") as opened:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        load_yaml(f"!include {path}", include_root=self.root)
                    opened.assert_not_called()

    def test_cycles_depth_and_total_bytes_are_bounded(self):
        (self.root / "a.yaml").write_text("!include b.yaml")
        (self.root / "b.yaml").write_text("!include a.yaml")
        with self.assertRaisesRegex(ValueError, "cycle"):
            load_yaml("!include a.yaml", include_root=self.root)
        with self.assertRaisesRegex(ValueError, "depth"):
            load_yaml("!include child.yaml", include_root=self.root, max_include_depth=0)
        with self.assertRaisesRegex(ValueError, "byte"):
            load_yaml("!include child.yaml", include_root=self.root, max_include_bytes=20)

    def test_internal_schemas_do_not_read_cwd_files(self):
        (self.root / "common.yaml").write_text("calibration: planted fixture\n")
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            schema = load_schema("lre")
        finally:
            os.chdir(previous)
        self.assertEqual(schema["calibration"]["type"], "dict")

    def test_read_yaml_requires_host_include_root(self):
        model_text = Path("dsge/examples/ar1/ar1.yaml").read_text()
        (self.root / "model.yaml").write_text(model_text)
        entry = self.root / "entry.yaml"
        entry.write_text("!include model.yaml")
        with self.assertRaisesRegex(ValueError, "disabled"):
            read_yaml(str(entry))
        model = read_yaml(str(entry), include_root=self.root)
        self.assertTrue(model)

    def test_root_stream_is_bounded_before_loading(self):
        class BoundedStream(io.StringIO):
            def read(self, size=-1):
                if size < 0 or size > 11:
                    raise AssertionError("unbounded root read")
                return super().read(size)

        with self.assertRaisesRegex(ValueError, "max_include_bytes"):
            read_yaml(BoundedStream("x" * 100), max_include_bytes=10)
        root_file = self.root / "large.yaml"
        root_file.write_text("x" * 100)
        with self.assertRaisesRegex(ValueError, "max_include_bytes"):
            read_yaml(str(root_file), max_include_bytes=10)
