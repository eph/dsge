import sympy
from sympy.printing.str import StrPrinter

from sympy.core.cache import clear_cache

reserved_names = {'log': sympy.log,
                  'exp': sympy.exp}
                  


clear_cache()

symbolic_context = {'__builtins__': None,
                    'log': sympy.log,
                    'exp': sympy.exp,
                    'sin': sympy.sin,
                    'cos': sympy.cos,
                    'tan': sympy.tan,
                    'asin': sympy.asin,
                    'acos': sympy.acos,
                    'atan': sympy.atan,
                    'sinh': sympy.sinh,
                    'cosh': sympy.cosh,
                    'tanh': sympy.tanh,
                    'sign': sympy.sign,
                    'sqrt': sympy.sqrt,
                    'normcdf': lambda x: 0.5 * (1 + erf(x / sqrt(2)))}

StrPrinter._print_TSymbol = lambda self, x: x.__str__()


class Parameter(sympy.Symbol):
    def __init__(self, name, exp_date=0):
        super(Parameter, self).__init__()
        self.name = name

    def __repr__(self):
        return self.name

    def __set_prior(self, prior):
        self.prior = prior


class TSymbol(sympy.Symbol):

    def __new__(cls, name, **args):
        obj = sympy.Symbol.__new__(cls, name)
        obj.date = args.get("date", 0)
        obj.exp_date = args.get("exp_date", 0)
        obj._mhash = None
        obj.__hash__()
        return obj

    def __call__(self, lead):
        newdate = int(self.date) + int(lead)
        newname = str(self.name)
        # print 'creating', newname, newdate
        clear_cache()
        return self.__class__(newname, date=newdate)

    # @property
    # def date(self):
    #     return self.assumptions0["date"]

    # @property
    # def exp_date(self):
    #     return self.assumptions0["exp_date"]

    def _hashable_content(self):
        return (self.name, str(self.date), str(self.exp_date))

    def __getstate__(self):
        return {
            "date": self.date,
            "name": self.name,
            "exp_date": self.exp_date,
            "is_commutative": self.is_commutative,
            "_mhash": self._mhash,
        }

    def class_key(self):
        return (2, 0, self.name, self.date)

    @property
    def lag(self):
        return self.date

    def __str__(self):
        if self.lag == 0:
            result = self.name
        else:
            result = self.name + r"(" + str(self.lag) + r")"
        return result

    # def __repr__(self):
    #     return self.__str__()

    # def _repr_(self):
    #     return self.__str__()

    # def repr(self):
    #     return self.__str__()

    # def _print_TSymbol(self):
    #     return self.__str__()

    # def _print(self):
    #     return self.__str__()


class Variable(TSymbol):
    @property
    def fortind(self):
        if self.date <= 0:
            return "v_" + self.name
        else:
            return "v_E" + self.name

    def __str__(self):
        if self.exp_date == 0:
            result = super(Variable, self).__str__()
        else:
            result = ("E[" + str(self.exp_date) + "]"
                      + super(Variable, self).__str__())

        return result

    def __repr__(self):
        return self.__str__()

    __sstr__ = __str__


class LaggedExpectation(Variable):
    def __init__(self, name, date=0, exp_date=0):
        Variable.__init__(self, name, date)
        self.exp_date = exp_date

    def __getstate_(self):
        return {
            "date": self.date,
            "name": self.name,
            "is_commutative": self.is_commutative,
            "_mhash": self._mhash,
        }

    def _hashable_content(self):
        return (self.name, self.date, self.lag)


class Shock(TSymbol):
    @property
    def fortind(self):
        if self.date <= 0:
            return "e_" + self.name
        else:
            return "e_E" + self.name


from typing import List

class Equation(sympy.Equality):
    """
    A class to represent an equation, which inherits from `sympy.Equality`.

    Attributes:
        lhs: The left-hand side of the equation
        rhs: The right-hand side of the equation
        name: The name of the equation (optional)
    """

    def __new__(cls, lhs: sympy.Expr, rhs: sympy.Expr, name: str = None) -> "Equation":
        return super(sympy.Equality, cls).__new__(cls, lhs, rhs)

    @property
    def set_eq_zero(self) -> sympy.Expr:
        """Returns the difference between the left-hand side and the right-hand side of the equation."""
        return self.lhs - self.rhs

    @property
    def variables(self) -> List[Variable]:
        """Returns a list of unique variables within the equation."""
        return [v for v in self.atoms() if isinstance(v, Variable)]
