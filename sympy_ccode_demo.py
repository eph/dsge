#!/home/eherbst/miniconda3/bin/python
"""
Demonstrates how to use sympy.ccode for C++ code generation with Eigen matrices.
"""

from sympy import symbols, Matrix, eye, zeros, exp, log
from sympy.printing.c import C99CodePrinter

class EigenMatrixPrinter(C99CodePrinter):
    """Custom C code printer for Eigen matrices"""
    
    def __init__(self):
        super().__init__()
        self.known_functions.update({
            'exp': 'stan::math::exp',
            'log': 'stan::math::log',
            'sqrt': 'stan::math::sqrt',
            'pow': 'stan::math::pow',
        })
    
    def _print_Pow(self, expr):
        if expr.exp.is_Number and float(expr.exp).is_integer():
            return super()._print_Pow(expr)
        else:
            return f"stan::math::pow({self._print(expr.base)}, {self._print(expr.exp)})"
    
    def _print_Matrix(self, expr):
        """Print matrices in C++ Eigen format"""
        rows, cols = expr.shape
        
        # For empty matrices
        if rows == 0 or cols == 0:
            return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>({rows}, {cols})"
        
        # For identity matrices or zero matrices, use specialized constructors
        if expr == eye(rows) and rows == cols:
            return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity({rows}, {cols})"
        
        if expr == zeros(rows, cols):
            return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero({rows}, {cols})"
        
        # For other matrices, use the comma initializer syntax
        elements = []
        for i in range(rows):
            row_elements = []
            for j in range(cols):
                element = expr[i, j]
                if element == 0:
                    row_elements.append("0.0")
                else:
                    row_elements.append(self._print(element))
            elements.append(", ".join(row_elements))
        
        matrix_code = "(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>({rows}, {cols}) << {elements}).finished()"
        return matrix_code.format(rows=rows, cols=cols, elements=", ".join(elements))

def main():
    """
    Demonstrates sympy.ccode with Eigen matrices.
    """
    # Define some symbols
    a, b, c = symbols('a b c')
    p1, p2, p3 = symbols('para[0] para[1] para[2]')
    
    # Create some example matrices
    m1 = Matrix([
        [p1 + p2, a * exp(p3), 0],
        [b * log(p1), p2 * p3, c],
        [p1**2, p2**0.5, p3**2]
    ])
    
    # Zero matrix
    m2 = zeros(3, 3)
    
    # Identity matrix
    m3 = eye(3)
    
    # Create our custom printer
    printer = EigenMatrixPrinter()
    
    # Generate C++ code
    print("// Example 1: Matrix with expressions")
    print("alpha0_cycle = " + printer._print_Matrix(m1) + ";\n")
    
    print("// Example 2: Zero matrix")
    print("alpha1_cycle = " + printer._print_Matrix(m2) + ";\n")
    
    print("// Example 3: Identity matrix")
    print("beta0_cycle = " + printer._print_Matrix(m3) + ";\n")
    
    # Show how expressions can be converted
    expr = a * exp(b + c) + log(p1 * p2) / p3**2
    print("// Example 4: Expression")
    print("double expr = " + printer.doprint(expr) + ";\n")
    
    print("// Example 5: Scalar parameter assignments")
    scalar_code = """
    // Parameter assignments
    const T& alpha = para(0);
    const T& beta = para(1);
    const T& gamma = para(2);
    const T& delta = para(3);
    
    // Derived parameter calculations
    T rho = stan::math::exp(alpha + beta);
    T phi = alpha / (1.0 - beta * gamma);
    """
    print(scalar_code)
    
    # Example of a complete matrix assignment with parameters
    full_example = """
    // A complete example of filling a matrix with parameters
    alpha0_cycle(0, 0) = 1.0;
    alpha0_cycle(0, 1) = -para(0);
    alpha0_cycle(0, 2) = 0.0;
    alpha0_cycle(1, 0) = 0.0;
    alpha0_cycle(1, 1) = 1.0;
    alpha0_cycle(1, 2) = -para(1);
    alpha0_cycle(2, 0) = -para(2) * stan::math::exp(para(3));
    alpha0_cycle(2, 1) = 0.0;
    alpha0_cycle(2, 2) = 1.0;
    """
    print("\n// Example 6: Element-wise matrix filling")
    print(full_example)

if __name__ == "__main__":
    main()