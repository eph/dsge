#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from dsge import read_yaml


def main() -> None:
    m = read_yaml("dsge/examples/fhp/fhp.yaml")
    p0 = m.p0()

    settings = {
        "mixed_pi10_hh2": {"default": 2, "by_lhs": {"pi": 10}},
        "scalar_k2": 2,
        "scalar_k10": 10,
        "scalar_k4": 4,
        "mixed_pi1": {"default": 4, "by_lhs": {"pi": 1}},
        "mixed_pi0": {"default": 4, "by_lhs": {"pi": 0}},
        "scalar_k0": 0,
    }

    shock = "e_mp"
    h = 10
    cols = ["pi", "c", "q"]

    out = {}
    for name, k in settings.items():
        cm = m.compile_model(k=k)
        out[name] = cm.impulse_response(p0, h)[shock][cols]

    for v in cols:
        tbl = pd.concat({k: df[v] for k, df in out.items()}, axis=1)
        print(f"\n{v} response to {shock} (t=0..5)")
        print(tbl.head(6).to_string())


if __name__ == "__main__":
    main()
