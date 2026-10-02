"""Keep executable CI inputs pinned and publication permissions scoped."""

from pathlib import Path
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; supplied by the locked dev dependencies.
    import tomli as tomllib

import yaml


ROOT = Path(__file__).resolve().parents[2]


class TestSupplyChainConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / ".github" / "workflows").glob("*.y*ml")
        }

    def test_external_actions_use_commit_shas(self):
        self.assertTrue(self.workflows)
        for name, workflow in self.workflows.items():
            for job in workflow["jobs"].values():
                for entry in [job, *job.get("steps", [])]:
                    action = entry.get("uses", "")
                    if action and not action.startswith("./"):
                        with self.subTest(workflow=name, action=action):
                            self.assertRegex(action, r"^[^@\s]+@[0-9a-f]{40}$")

    def test_uv_installation_is_versioned_and_verified(self):
        for name, workflow in self.workflows.items():
            for job in workflow["jobs"].values():
                for step in job.get("steps", []):
                    if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                        with self.subTest(workflow=name):
                            options = step["with"]
                            self.assertRegex(options["version"], r"^\d+\.\d+\.\d+$")
                            self.assertRegex(options["checksum"], r"^[0-9a-f]{64}$")

    def test_workflow_dependencies_use_checked_frozen_lock(self):
        for name, workflow in self.workflows.items():
            for job in workflow["jobs"].values():
                commands = "\n".join(
                    step.get("run", "") for step in job.get("steps", [])
                )
                with self.subTest(workflow=name):
                    self.assertNotRegex(commands, r"\bpip\s+install\b")
                    if "uv run" in commands:
                        bootstrap = "uv sync --frozen --group build --no-install-project"
                        check = "uv lock --check --no-build-isolation-package dsge"
                        install = "uv sync --frozen --group build --no-build-isolation-package dsge"
                        self.assertIn(bootstrap, commands)
                        self.assertIn(check, commands)
                        self.assertIn(install, commands)
                        self.assertLess(commands.index(bootstrap), commands.index(check))
                        self.assertLess(commands.index(check), commands.index(install))
                        for line in commands.splitlines():
                            if "uv run" in line:
                                self.assertIn("uv run --frozen --no-sync", line)

    def test_build_tools_are_in_the_lock(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
        locked = {package["name"] for package in packages}

        def package_name(requirement):
            return re.match(r"[\w-]+", requirement).group().lower().replace("_", "-")

        available = {
            package_name(requirement)
            for requirement in (
                config["project"]["dependencies"] + config["dependency-groups"]["build"]
            )
        }
        for requirement in config["build-system"]["requires"]:
            with self.subTest(requirement=requirement):
                name = package_name(requirement)
                self.assertIn(name, available)
                self.assertIn(name, locked)

    def test_pages_and_oidc_permissions_only_reach_deploy(self):
        docs = self.workflows["docs.yml"]
        for name, job in docs["jobs"].items():
            permissions = job.get("permissions", docs.get("permissions", {}))
            with self.subTest(job=name):
                if name == "deploy":
                    self.assertEqual(permissions.get("pages"), "write")
                    self.assertEqual(permissions.get("id-token"), "write")
                else:
                    self.assertEqual(permissions.get("contents"), "read")
                    self.assertNotEqual(permissions.get("pages"), "write")
                    self.assertNotEqual(permissions.get("id-token"), "write")

    def test_git_dependencies_match_locked_commits(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
        requirements = list(project["dependencies"])
        for extras in project.get("optional-dependencies", {}).values():
            requirements.extend(extras)
        for requirement in requirements:
            if "git+" not in requirement:
                continue
            with self.subTest(requirement=requirement):
                match = re.fullmatch(
                    r"([\w-]+)\s*@\s*git\+(https://[^@\s]+)@([0-9a-f]{40})",
                    requirement,
                )
                self.assertIsNotNone(match, "Git dependencies must name a full commit")
                name, url, commit = match.groups()
                locked = next(package for package in packages if package["name"] == name)
                self.assertEqual(locked["source"]["git"], f"{url}?rev={commit}#{commit}")

    def test_setup_instructions_do_not_execute_remote_scripts(self):
        instructions = (ROOT / ".beads" / "README.md").read_text()
        self.assertNotRegex(
            instructions,
            r"(?:curl|wget)[^\n]*\|\s*(?:ba)?sh\b",
        )
        self.assertNotRegex(
            instructions,
            r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/(?:main|master)/",
        )


if __name__ == "__main__":
    unittest.main()
