from unittest import TestCase

from dsge.dynare_parser import parse_mod_text
from dsge.dynare_translate import to_yaml_like


class TestDynareImport(TestCase):
    def test_ignores_equation_labels_and_commands(self):
        mod_text = """
var x y;
varexo e;
parameters a;
a = 0.9;

model;
[name='eq_x']
x = a*x(-1) + e;
[name='eq_y']
y = x(+8);
end;

stoch_simul(
  order=1
);
""".strip()

        parsed = parse_mod_text(mod_text)
        self.assertEqual(parsed["variables"], ["x", "y"])
        self.assertEqual(parsed["shocks"], ["e"])
        self.assertEqual(parsed["parameters"], ["a"])
        self.assertEqual([eq.strip() for eq in parsed["equations"]], ["x = a*x(-1) + e", "y = x(+8)"])

        yaml_like = to_yaml_like(parsed, name="tiny")
        self.assertEqual(yaml_like["declarations"]["max_lag"], 1)
        self.assertEqual(yaml_like["declarations"]["max_lead"], 8)

