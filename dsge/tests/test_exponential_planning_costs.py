import copy
import io
import math
import unittest
from importlib.resources import files

import numpy as np
import yaml
from numpy.testing import assert_allclose

from dsge import ExponentialMarginalCostSchedule, LinearMarginalCostSchedule, read_yaml
from dsge.endogenous_horizon_switching import choose_k_star
from dsge.parse_yaml import ValidationError


EXAMPLE = files("dsge") / "examples/fhp/nk_endogenous_exponential_costs.yaml"


def switching_spec():
    # Policy increments equal x, so MB = x**2/2 independently of k.
    return yaml.safe_load("""
declarations:
  name: exponential_cost_test
  type: switching_ssm
  components: [planner]
  states: [x]
  shocks: [e]
  observables: [y]
  parameters: [cost_level, cost_growth]
  horizon_choice:
    components:
      planner:
        k_max: 8
        cost: {type: exponential, a: cost_level, growth: cost_growth}
        lambda: 1
        policy_object: y
model:
  TT: [[1]]
  RR: [[1]]
  ZZ: [[k_planner + 1]]
  DD: [0]
  QQ: [[0]]
  HH: [[0]]
calibration:
  parameters: {cost_level: 0.1, cost_growth: 0.6931471805599453}
""")


def load(spec):
    return read_yaml(io.StringIO(yaml.safe_dump(spec, sort_keys=False)))


class TestExponentialPlanningCosts(unittest.TestCase):
    def test_schedule_indexing_flat_limit_and_overflow(self):
        cost = ExponentialMarginalCostSchedule(0.1, math.log(2))
        assert_allclose([cost.delta_tau(j) for j in range(1, 5)], [.1, .2, .4, .8])
        for j in (1, 2, 1000):
            self.assertEqual(ExponentialMarginalCostSchedule(.1, 0).delta_tau(j), .1)
        self.assertAlmostEqual(
            ExponentialMarginalCostSchedule(1e-300, 710).delta_tau(2),
            math.exp(math.log(1e-300) + 710),
        )
        self.assertEqual(ExponentialMarginalCostSchedule(1, 1000).delta_tau(2), math.inf)
        with self.assertRaises(ValueError):
            cost.delta_tau(0)
        for a, growth in [(0, .1), (-1, .1), (math.inf, .1), (1, -.1),
                          (1, math.nan), (1, math.inf)]:
            with self.subTest(a=a, growth=growth), self.assertRaises(ValueError):
                ExponentialMarginalCostSchedule(a, growth)

    def test_strict_tie_and_nonfinite_benefits(self):
        args = dict(params=np.array([]), info_t=None, component="planner", k_max=3)
        # Equality at j=1 continues; j=2 is the first rejected stage.
        self.assertEqual(choose_k_star(
            **args, mb=lambda *args: .5,
            cost=ExponentialMarginalCostSchedule(.5, math.log(2)),
        ), 1)
        for value in (math.inf, math.nan, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                choose_k_star(**args, mb=lambda *args: value,
                              cost=ExponentialMarginalCostSchedule(1, 1000))

    def test_switching_selection_simulation_girf_and_parameter_cache(self):
        model = load(switching_spec())
        params = np.array(model.p0, copy=True)
        for growth, level, horizon in [(math.log(2), .1, 3), (math.log(4), .1, 2),
                                       (math.log(2), .3, 1), (math.log(2), .1, 3)]:
            params[model.parameter_names.index("cost_growth")] = growth
            params[model.parameter_names.index("cost_level")] = level
            self.assertEqual(model.choose_regime(params, np.array([1.]), t=0), (horizon,))
            sim = model.simulate(params, 3, x0=np.array([1.]), seed=42)
            assert_allclose(sim["s_path"], horizon)
            assert_allclose(sim["y_path"], horizon + 1)
            irf = model.girf(params, h=3, reps=1, shock_size=1, seed=42)
            assert_allclose(irf["girf"].to_numpy().ravel(), [0, horizon + 1, horizon + 1])
            assert_allclose(irf["k_shocked_mean"].to_numpy().ravel(), [0, horizon, horizon])

    def test_python_cost_inputs_remain_compatible(self):
        model = load(switching_spec())
        for raw, horizon in [(.1, 8), ((.1, .2), 2),
                             (LinearMarginalCostSchedule(.1, .2), 2),
                             (ExponentialMarginalCostSchedule(.1, math.log(2)), 3)]:
            with self.subTest(raw=raw):
                model._cost_func = lambda params, component: raw
                model._cost_cache.clear()
                self.assertEqual(model.choose_regime(model.p0, np.array([1.]), t=0), (horizon,))

    def test_yaml_validation_in_both_loaders(self):
        for original in (switching_spec(), yaml.safe_load(EXAMPLE.read_text())):
            key = "stopping_rule" if original["declarations"]["type"] == "fhp" else "horizon_choice"
            component = next(iter(original["declarations"][key]["components"]))
            for cost in [dict(type="exponential", a=.1),
                         dict(type="exponential", a=.1, growth=.2, b=.1),
                         dict(a=.1, growth=.2),
                         dict(type="exponentail", a=.1, growth=.2),
                         dict(type="exponential", a=.1, growth="unknown_parameter"),
                         dict(type="exponential", a=.1, growth=-1),
                         dict(type="exponential", a=0, growth=.2)]:
                spec = copy.deepcopy(original)
                spec["declarations"][key]["components"][component]["cost"] = cost
                with self.subTest(kind=spec["declarations"]["type"], cost=cost):
                    with self.assertRaises((ValueError, ValidationError)):
                        load(spec)
        model = load(switching_spec())
        params = np.array(model.p0, copy=True)
        params[model.parameter_names.index("cost_growth")] = -.1
        with self.assertRaisesRegex(ValueError, "growth"):
            model.choose_regime(params, np.array([1.]), t=0)

    def test_flat_limit_and_linear_yaml_in_both_loaders(self):
        for original in (switching_spec(), yaml.safe_load(EXAMPLE.read_text())):
            key = "stopping_rule" if original["declarations"]["type"] == "fhp" else "horizon_choice"
            outputs = []
            for cost in [dict(a=.1), dict(type="exponential", a=.1, growth=0),
                         dict(type="linear", a=.1, b=0)]:
                spec = copy.deepcopy(original)
                for cfg in spec["declarations"][key]["components"].values():
                    cfg["cost"] = cost
                    cfg["k_max"] = 4
                if key == "stopping_rule":
                    # Exercise the FHP alias and simultaneous choice too.
                    spec["declarations"]["horizon_choice"] = spec["declarations"].pop(key)
                    spec["declarations"]["horizon_choice"]["selection_mode"] = "simultaneous"
                model = load(spec)
                outputs.append(model.simulate(model.p0, 3, x0=np.array([1.]), seed=123))
            for output in outputs[1:]:
                for field in ("s_path", "y_path", "x_path"):
                    assert_allclose(output[field], outputs[0][field], atol=1e-12)
        spec = switching_spec()
        spec["declarations"]["horizon_choice"]["components"]["planner"]["cost"] = {
            "type": "linear", "a": "cost_level", "b": "cost_growth"}
        spec["calibration"]["parameters"]["cost_growth"] = .2
        model = load(spec)
        self.assertEqual(model.choose_regime(model.p0, np.array([1.]), t=0), (2,))

    def test_native_ge_threshold_matches_independent_recursion(self):
        model = read_yaml(str(EXAMPLE))
        params = np.array(model.p0, copy=True)
        cut = .0025
        matrix = np.array([[1., 1.], [.024, 1.014]])
        impact = np.array([1., .024]) * cut
        levels = np.array([1e-6, 1e-9])
        for growth, expected in [(.29, (59, 59)), (.31, (28, 59)), (.29, (59, 59))]:
            params[model.parameter_names.index("cost_growth")] = growth
            increments = impact.copy()
            horizons = [59, 59]
            common = [impact.copy()]
            for j in range(1, 60):
                increments = matrix @ increments
                common.append(common[-1] + increments)
                for c in range(2):
                    if horizons[c] == 59 and .5 * increments[c]**2 < levels[c]*math.exp(growth*(j-1)):
                        horizons[c] = j-1
            self.assertEqual(tuple(horizons), expected)
            chosen = model.choose_regime(params, np.array([cut]), t=0)
            self.assertEqual(chosen, expected)
            h, f = horizons
            y = cut + common[h-1][0] + common[h-1][1]
            pi = .024*y + .99*common[f-1][1]
            _, _, zz, dd, _, _ = model.get_mats(params, chosen)
            assert_allclose(zz @ np.array([cut]) + dd, [y, pi], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
