import sympy
import re
from sympy.printing.str import StrPrinter
from sympy.core.cache import clear_cache
from sympy import Function

reserved_names = {'log': sympy.log,
                  'exp': sympy.exp}

# Greek letter mapping for automatic Unicode substitution
GREEK_LETTERS = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ', 'Epsilon': 'Ε',
    'Zeta': 'Ζ', 'Eta': 'Η', 'Theta': 'Θ', 'Iota': 'Ι', 'Kappa': 'Κ',
    'Lambda': 'Λ', 'Mu': 'Μ', 'Nu': 'Ν', 'Xi': 'Ξ', 'Omicron': 'Ο',
    'Pi': 'Π', 'Rho': 'Ρ', 'Sigma': 'Σ', 'Tau': 'Τ', 'Upsilon': 'Υ',
    'Phi': 'Φ', 'Chi': 'Χ', 'Psi': 'Ψ', 'Omega': 'Ω'
}

# Common economic variable substitutions
ECON_SYMBOLS = {
    'rho': 'ρ', 'sigma': 'σ', 'theta': 'θ', 'phi': 'φ', 'delta': 'δ',
    'beta': 'β', 'gamma': 'γ', 'lambda': 'λ', 'epsilon': 'ε', 'pi': 'π'
}

def convert_to_greek(name):
    """Convert variable names to Greek letters where appropriate."""
    # Exact match for full name
    if name.lower() in GREEK_LETTERS:
        return GREEK_LETTERS[name.lower()]
    
    # Common prefixes in economic variables
    for greek, symbol in ECON_SYMBOLS.items():
        # Only convert if it's a standalone prefix (e.g., 'beta' but not 'betahat')
        # or if it's a prefix followed by underscore or digit
        if name == greek or re.match(f"^{greek}_", name) or re.match(f"^{greek}[0-9]", name):
            return name.replace(greek, symbol, 1)
    
    return name

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
        # Convert parameter names to Greek letters where appropriate
        return convert_to_greek(self.name)

    def __str__(self, greek=False):
        # Use Greek letter representation for string output too
        if not greek: return self.name
        return convert_to_greek(self.name)

    def __set_prior(self, prior):
        self.prior = prior


class TSymbol(sympy.Symbol):

    def __new__(cls, name, **args):
        clear_cache()
        obj = sympy.Symbol.__new__(cls, name)
        obj.date = args.get("date", 0)
        obj.exp_date = args.get("exp_date", 0)
        obj._mhash = None
        obj.__hash__()
        return obj

    def __call__(self, lead):
        newdate = int(self.date) + int(lead)
        newname = str(self.name)
        clear_cache()
        return self.__class__(newname, date=newdate)

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

    def __repr__(self):
        return self.__str__(greek=True)

    def __str__(self, greek=False):
        # Convert variable name to Greek if appropriate

        greek_name = convert_to_greek(self.name)
        
        if self.lag == 0:
            result = greek_name
        elif self.lag > 0:
            # Future value: use subscript t+n notation
            result = f"{greek_name}_{{+{self.lag}}}"
        else:
            # Past value: use subscript t-n notation
            lag_abs = abs(self.lag)
            result = f"{greek_name}_{{-{lag_abs}}}"
            
        return result


class Variable(TSymbol):
    
    def __repr__(self):
        return self.__str__(greek=True)

    def __str__(self, greek=False):
        # Convert variable name to Greek if appropriate
        if not greek:
            if self.lag == 0:
                return self.name
            else:
                return self.name + r"(" + str(self.lag) + r")"
        greek_name = convert_to_greek(self.name)
        
        if self.lag == 0:
            # Current period
            base_result = f"{greek_name}_t"
        elif self.lag > 0:
            # Future period
            base_result = f"{greek_name}_{{t+{self.lag}}}"
        else:
            # Past period
            lag_abs = abs(self.lag)
            base_result = f"{greek_name}_{{t-{lag_abs}}}"
        
        if self.exp_date == 0:
            # No expectation
            result = base_result
        else:
            # With expectation
            # Format as: E_t[x_{t+1}]
            result = f"𝔼_t[{base_result}]"
            
        return result

    __sstr__ = __str__


class LaggedExpectation(Variable):
    def __new__(cls, name, date=0, exp_date=0):
        obj = Variable.__new__(cls, name, date=date)
        obj.exp_date = exp_date
        return obj
        
    def __init__(self, name, date=0, exp_date=0):
        # No need to call parent __init__ as we're using __new__
        pass

    def __getstate_(self):
        return {
            "date": self.date,
            "name": self.name,
            "exp_date": self.exp_date,
            "is_commutative": self.is_commutative,
            "_mhash": self._mhash,
        }

    def _hashable_content(self):
        return (self.name, self.date, self.lag)
    
    def __str__(self):
        # Convert variable name to Greek if appropriate
        greek_name = convert_to_greek(self.name)
        
        if self.lag == 0:
            # Current period
            base_result = f"{greek_name}_t"
        elif self.lag > 0:
            # Future period
            base_result = f"{greek_name}_{{t+{self.lag}}}"
        else:
            # Past period
            lag_abs = abs(self.lag)
            base_result = f"{greek_name}_{{t-{lag_abs}}}"
            
        # Format as: E_{t-j}[x_{t+k}] for expectation formed at t-j
        return f"𝔼_{{t-{self.exp_date}}}[{base_result}]"
        
    def __repr__(self):
        return self.__str__()


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
        obj = super(sympy.Equality, cls).__new__(cls, lhs, rhs)
        obj.name = name
        return obj

    @property
    def set_eq_zero(self) -> sympy.Expr:
        """Returns the difference between the left-hand side and the right-hand side of the equation."""
        return self.lhs - self.rhs

    @property
    def variables(self) -> List[Variable]:
        """Returns a list of unique variables within the equation."""
        return [v for v in self.atoms() if isinstance(v, Variable)]
    
    def format_expression(self, expr):
        """Format expression with proper Greek letters for parameters and mathematical notation."""
        # Convert to string first
        expr_str = str(expr)
        
        # Find all parameter names and replace them with Greek letters
        for atom in self.atoms():
            if isinstance(atom, Parameter):
                # Make sure we replace whole words with word boundaries
                # This regex replaces 'alpha' but not 'alphabet'
                parameter_name_pattern = r'\b' + atom.name + r'\b'
                greek_symbol = convert_to_greek(atom.name)
                expr_str = re.sub(parameter_name_pattern, greek_symbol, expr_str)
        
        # Pattern that matches variable**2 or parameter**2 (must come before replacing * with ·)
        expr_str = re.sub(r'(\w+)\s*\*\*\s*2', r'\1²', expr_str)
        
        # Replace * with · for multiplication - but not in the ** for powers
        expr_str = re.sub(r'(?<!\*)\*(?!\*)', '·', expr_str)
        
        # Pattern that matches variable**N for any N
        superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
        
        def replace_superscript(match):
            exponent = match.group(1)
            return ''.join(superscripts.get(c, c) for c in exponent)
            
        expr_str = re.sub(r'\*\*(\d+)', replace_superscript, expr_str)
        
        return expr_str
    
    def __str__(self) -> str:
        """Enhanced string representation with proper formatting and Greek letters."""
        lhs_str = self.format_expression(self.lhs)
        rhs_str = self.format_expression(self.rhs)
        return f"{lhs_str} = {rhs_str}"
    
    def __repr__(self) -> str:
        """Enhanced representation with equation name if available."""
        if self.name:
            return f"{self.name}: {self.__str__()}"
        return self.__str__()

class EXP(Function):

    @classmethod
    def eval(cls, j, x):
        if len(x.atoms()) > 1:                                          
            x = x.subs({xa:EXP(j)(xa) for xa in x.atoms() if isinstance(xa, Variable)})
            return x   
        elif isinstance(x, Variable):
            return Variable(x.name, date=x.date, exp_date=j)
        elif isinstance(x, Parameter):
            return x
        else:
            return None

    def __new__(cls, j):
        return lambda x: EXP.eval(j, x)

