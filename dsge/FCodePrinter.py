from sympy.printing.fcode import FCodePrinter
from sympy.printing.codeprinter import Assignment


def _print_Assignment(self, expr):
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.indexed import IndexedBase
    lhs = expr.lhs
    rhs = expr.rhs
    # We special case assignments that take multiple lines
    if isinstance(expr.rhs, Piecewise):
        # Here we modify Piecewise so each expression is now
        # an Assignment, and then continue on the print.
        expressions = []
        conditions = []
        for (e, c) in rhs.args:
            expressions.append(Assignment(lhs, e))
            conditions.append(c)
        temp = Piecewise(*zip(expressions, conditions))
        return self._print(temp)
    elif isinstance(lhs, MatrixSymbol):
        # Here we form an Assignment for each element in the array,
        # printing each one.
        lines = []
        for (i, j) in self._traverse_matrix_indices(lhs):
            temp = Assignment(lhs[i, j], rhs[i, j])
            if not(rhs[i, j] == 0):
                code0 = self._print(temp)
                lines.append(code0)

        return "\n".join(lines)
    elif self._settings["contract"] and (lhs.has(IndexedBase) or
            rhs.has(IndexedBase)):
        # Here we check if there is looping to be done, and if so
        # print the required loops.
        return self._doprint_loops(rhs, lhs)
    else:
        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        return self._get_statement("%s = %s" % (lhs_code, rhs_code))

FCodePrinter._print_Assignment = _print_Assignment

from sympy.printing.fcode import fcode

import sympy 





def fcode_double(x, assign_to=None, **settings):
    
    if isinstance(x, (int, float)):
        return fcode(sympy.Float(x), assign_to=assign_to, **settings)
    else:
        return fcode(x.subs([(si, sympy.Float(si)) for si in x.atoms(sympy.Integer)]), assign_to=assign_to, **settings)