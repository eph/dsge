from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from lark import Lark, Transformer, v_args

GRAMMAR_PATH = Path(__file__).with_name("dynare.lark")


def _load_parser() -> Lark:
    with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
        grammar = f.read()
    return Lark(grammar, start="start", parser="lalr")


@v_args(inline=True)
class _DynareTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.variables: List[str] = []
        self.varexo: List[str] = []
        self.parameters: List[str] = []
        self.param_values: Dict[str, str] = {}
        self.equations: List[str] = []
        self.covariance: Dict[str, str] = {}
        self.correlations: List[Tuple[str, str, str]] = []
        self.initval: Dict[str, str] = {}
        self.endval: Dict[str, str] = {}
        self.varexo_det: List[str] = []
        self.predetermined: List[str] = []
        self.observables: List[str] = []
        self._last_shock_name: str | None = None

    # name_list returns list of tokens; convert to python strings
    def name_list(self, *names):
        return [str(n) for n in names]

    def var_block(self, names):
        self.variables.extend(names)
        return None

    def varexo_block(self, names):
        self.varexo.extend(names)
        return None

    def varexo_det_block(self, names):
        self.varexo_det.extend(names)
        return None

    def parameters_block(self, names):
        self.parameters.extend(names)
        return None

    def predetermined_block(self, names):
        self.predetermined.extend(names)
        return None

    def varobs_block(self, names):
        self.observables = names
        return None

    def param_assign(self, name, expr):
        self.param_values[str(name)] = str(expr).strip()
        return None

    def equation(self, content):
        self.equations.append(str(content).strip())
        return None

    def expr(self, content):
        # Ensure expressions are plain strings, not Lark Trees/Tokens
        return str(content)

    def shock_var(self, name, expr=None):
        # Dynare: var e = 0.01^2; or just var e; (we only map when a value is provided)
        self._last_shock_name = str(name)
        if expr is not None:
            self.covariance[str(name)] = str(expr).strip()
        return None

    def shock_stderr(self, name, expr):
        # Map stderr sigma to variance sigma^2 in calibration
        ex = str(expr).strip()
        self.covariance[str(name)] = f"({ex})**2"
        return None

    def shock_stderr_bare(self, expr):
        # Dynare allows `var e; stderr 0.1;` meaning last declared var
        if not self._last_shock_name:
            raise ValueError("stderr without a preceding `var <name>;` in shocks block")
        ex = str(expr).strip()
        self.covariance[self._last_shock_name] = f"({ex})**2"
        return None

    def shock_corr(self, n1, n2, expr):
        self.correlations.append((str(n1), str(n2), str(expr).strip()))
        return None

    def assign_stmt(self, name, expr):
        return (str(name), str(expr).strip())

    def initval_block(self, *assigns):
        for name, expr in assigns:
            self.initval[name] = expr
        return None

    def endval_block(self, *assigns):
        for name, expr in assigns:
            self.endval[name] = expr
        return None


def parse_mod_text(text: str) -> Dict[str, Any]:
    parser = _load_parser()
    tree = parser.parse(text)
    t = _DynareTransformer()
    t.transform(tree)

    # Deduplicate while preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "variables": dedup(t.variables),
        "shocks": dedup(t.varexo),
        "parameters": dedup(t.parameters),
        "equations": t.equations,
        "param_values": t.param_values,
        "covariance": t.covariance,
        "initval": t.initval,
        "endval": t.endval,
        "observables": t.observables,
        "correlations": t.correlations,
    }


def parse_mod_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    return parse_mod_text(txt)
