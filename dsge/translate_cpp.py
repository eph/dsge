#!/usr/bin/env python3
"""
Translate DSGE models to C++ code.

This module provides functions to translate DSGE models to C++ code,
leveraging Eigen matrices and stan-math types for automatic differentiation.
"""

import os
import numpy as np
import sympy
from .symbols import Parameter
from .parsing_tools import parse_expression

# C++ template for the FHP model
cpp_model = """
/**
 * @file model_t.hpp
 * @brief C++ implementation of an FHP DSGE model
 * 
 * This file contains the C++ implementation of an FHP DSGE model,
 * using Eigen matrices and stan-math for automatic differentiation.
 */

#ifndef MODEL_T_HPP
#define MODEL_T_HPP

#include <Eigen/Dense>
#include <stan/math.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <boost/random.hpp>

template <typename T1, typename T2, typename T3>
typename stan::return_type<T1, T2, T3>::type ig_lpdf(const T1 x, const T2 a,
                                                     const T3 b) {{
  using ReturnType = typename stan::return_type<T1, T2, T3>::type;
  ReturnType logP = stan::math::log(2.0) - stan::math::lgamma(b / 2.0) +
                    b / 2.0 * stan::math::log(b * a * a / 2.0) -
                    (b + 1.0) / 2.0 * stan::math::log(x * x) -
                    b * a * a / (2.0 * x * x);
  return logP;
}}

double ig_rng(const double a, const int b, boost::random::mt19937 &rng) {{
  double rvs = 0;

  for (int i = 0; i < b; i++) {{
    double draw = stan::math::normal_rng(0.0, 1.0, rng);
    rvs += draw * draw;
  }}
  rvs = stan::math::sqrt(b * a * a / rvs);
  return rvs;
}}


/**
 * @class Model
 * @brief Implements an FHP DSGE model
 */
class Model {{
public:
    // Model dimensions
    int nvar;     // Number of variables
    int nshock;   // Number of shocks
    int nval;     // Number of values
    int nobs;     // Number of observables
    int npara;    // Number of parameters
    int neps;     // Number of innovations
    int ns;       // Total number of states
    int k;        // Number of iterations
    int t0;       // Starting time 

    // Constructor
    Model() 
    : nvar({var_count}), 
      nshock({shock_count}), 
      nval({val_count}),
      nobs({obs_count}),
      npara({para_count}),
      neps({eps_count}),
      k({k_val}),
      t0({t0}) 
    {{
        ns = 3 * nvar + nval + nshock;
    }}

    /**
     * Computes system matrices for the state space model
     *
     * @param para Vector of parameters
     * @param TT Transition matrix
     * @param RR Shock impact matrix
     * @param QQ Shock covariance matrix
     * @param DD Measurement equation constant vector
     * @param ZZ Measurement matrix
     * @param HH Measurement error covariance matrix
     * @return Error code (0 for success)
     */
    template <typename T>
    int system_matrices(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& para,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& TT,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& RR,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& QQ,
        Eigen::Matrix<T, Eigen::Dynamic, 1>& DD,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& ZZ,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& HH
    ) {{
        // Fill matrices based on parameters
        {sims_mat}

        // Initial calculations for A_cycle, B_cycle, A_trend, B_trend
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_cycle = alpha0_cycle.inverse() * alpha1_cycle;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_cycle = alpha0_cycle.inverse() * beta0_cycle;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_trend = alpha0_trend.inverse() * alpha1_trend;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_trend = alpha0_trend.inverse() * betaV_trend;

        // Main loop for k iterations
        for (int i = 1; i <= k; i++) {{
            // Calculations for A_cycle_new
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp1 = alphaC_cycle - alphaF_cycle * A_cycle;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp1_inv = temp1.inverse();
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_cycle_new = temp1_inv * alphaB_cycle;
            
            // Calculations for B_cycle_new
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_cycle_new = temp1_inv * (alphaF_cycle * B_cycle * P + betaS_cycle);
            
            // Calculations for A_trend_new
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp2 = alphaC_trend - alphaF_trend * A_trend;
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp2_inv = temp2.inverse();
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_trend_new = temp2_inv * alphaB_trend;
            
            // Calculations for B_trend_new
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_trend_new = temp2_inv * (alphaF_trend * B_trend);
            
            // Update matrices
            A_cycle = A_cycle_new;
            B_cycle = B_cycle_new;
            A_trend = A_trend_new;
            B_trend = B_trend_new;
        }}

        // Initialize transition matrix TT and shock matrix RR
        TT.setZero(ns, ns);
        RR.setZero(ns, neps);
        
        // Fill transition matrix TT
        // First block row
        TT.block(0, 0, nvar, nvar) = B_trend * value_gamma * value_Cx;
        TT.block(0, nvar, nvar, nvar) = A_cycle;
        TT.block(0, 2*nvar, nvar, nvar) = A_trend;
        TT.block(0, 3*nvar, nvar, nval) = B_trend * value_gammaC;
        TT.block(0, 3*nvar+nval, nvar, nshock) = B_cycle * P + B_trend * value_gamma * value_Cs;
        
        // Second block row
        TT.block(nvar, nvar, nvar, nvar) = A_cycle;
        TT.block(nvar, 3*nvar+nval, nvar, nshock) = B_cycle * P;
        
        // Third block row
        TT.block(2*nvar, 0, nvar, nvar) = B_trend * value_gamma * value_Cx;
        TT.block(2*nvar, 2*nvar, nvar, nvar) = A_trend;
        TT.block(2*nvar, 3*nvar, nvar, nval) = B_trend * value_gammaC;
        TT.block(2*nvar, 3*nvar+nval, nvar, nshock) = B_trend * value_gamma * value_Cs;
        
        // Fourth block row
        TT.block(3*nvar, 0, nval, nvar) = value_gamma * value_Cx;
        TT.block(3*nvar, 3*nvar, nval, nval) = value_gammaC;
        TT.block(3*nvar, 3*nvar+nval, nval, nshock) = value_gamma * value_Cs;
        
        // Fifth block row
        TT.block(3*nvar+nval, 3*nvar+nval, nshock, nshock) = P;
        
        // Fill shock matrix RR
        RR.block(0, 0, nvar, neps) = B_cycle * R;
        RR.block(nvar, 0, nvar, neps) = B_cycle * R;
        RR.block(3*nvar+nval, 0, nshock, neps) = R;
        
        // Set measurement equation
        DD = DD2;
        
        return 0; // Success
    }}

    /**
     * Load data from a file
     *
     * @param filename The name of the file to load
     * @return Matrix containing the data
     */
    Eigen::MatrixXd load_data(const std::string& filename) {{
        std::ifstream file(filename);
        if (!file.is_open()) {{
            throw std::runtime_error("Failed to open data file: " + filename);
        }}
        
        std::vector<std::vector<double>> data;
        std::string line;
        
        while (std::getline(file, line)) {{
            std::istringstream iss(line);

            std::vector<double> row;
            double value;
            
            while (iss >> value) {{
                row.push_back(value);
            }}
            
            if (!row.empty()) {{
                data.push_back(row);
            }}
        }}
        
        if (data.empty()) {{
            throw std::runtime_error("No data found in file: " + filename);
        }}
        
        int rows = data.size();
        int cols = data[0].size();
        
        Eigen::MatrixXd result(rows, cols);
        for (int i = 0; i < rows; i++) {{
            for (int j = 0; j < cols; j++) {{
                result(i, j) = data[i][j];
            }}
        }}
        
        return result;
    }}



}};
{log_prior}

{prior_draws}
#endif // MODEL_T_HPP
"""

import re

def generate_dsge_logprior(prior_list):
    """
    Generate C++ code for a dsge_logprior function that sums up
    the log-likelihood contributions from each prior using Stan Math.
    
    Parameters
    ----------
    prior_list : list
        A list of SciPy frozen distributions, e.g. stats.norm(loc=..., scale=...),
        stats.gamma(a=..., scale=...), etc.
        
    Returns
    -------
    str
        C++ code for dsge_logprior.
    """
    # We’ll build up lines of code and then return a single string.
    code_lines = []
    
    code_lines.append("template <typename T>")
    code_lines.append("T dsge_logprior(const std::vector<T>& para) {")
    code_lines.append("    using stan::math::normal_lpdf;")
    code_lines.append("    using stan::math::gamma_lpdf;")
    code_lines.append("    using stan::math::beta_lpdf;")
    code_lines.append("    using stan::math::uniform_lpdf;")
    code_lines.append("    using stan::math::inv_gamma_lpdf;")
    # ... add other 'using' statements for distributions you anticipate.
    
    code_lines.append("    T lp = 0.0;")
    
    for i, rv in enumerate(prior_list):
        # The distribution name might be something like 'norm', 'gamma', 'beta', etc.
        dist_name = rv.dist.name.lower()  # e.g. 'norm'
        
        # Each SciPy frozen distribution has parameters accessible in different ways.
        # For example, a normal has rv.kwds['loc'], rv.kwds['scale'].
        #
        # We'll handle some typical ones below. You can expand as needed.
        
        if dist_name == 'norm':
            mu = rv.kwds['loc']
            sigma = rv.kwds['scale']
            code_lines.append(f"    lp += normal_lpdf(para[{i}], T({mu}), T({sigma}));")
        
        elif dist_name == 'gamma':
            # In SciPy, gamma(a, loc=0, scale=1) has shape parameter a,
            # but Stan's gamma_lpdf is parameterized by alpha, beta (rate!).
            # Alternatively you can do alpha = a, beta = 1/scale.
            # So if in SciPy you used gamma(a=5, scale=2),
            # that implies shape=5, scale=2 => rate=1/2
            print(rv.kwds)
            shape = rv.args[0]
            scale = rv.kwds.get('scale', 1.0)
            rate = 1.0 / scale
            code_lines.append(f"    lp += gamma_lpdf(para[{i}], T({shape}), T({rate}));")
        
        elif dist_name == 'beta':
            # For SciPy: beta(a, b, loc=0, scale=1)
            # For Stan: beta_lpdf(x| alpha, beta).
            alpha = rv.args[0]
            beta = rv.args[1]
            code_lines.append(f"    lp += beta_lpdf(para[{i}], T({alpha}), T({beta}));")
        
        elif dist_name == 'uniform':
            # SciPy: uniform(loc, scale)
            # => it’s uniform over [loc, loc+scale]
            low = rv.kwds['loc']
            high = low + rv.kwds['scale']
            # Stan’s uniform_lpdf is uniform_lpdf(x| low, high)
            code_lines.append(f"    lp += uniform_lpdf(para[{i}], T({low}), T({high}));")
        
        elif dist_name == 'invgamma_zellner':
            # SciPy: invgamma(a, loc=0, scale=1)
            # shape = a, scale = scale => Stan's inv_gamma_lpdf(y | alpha, beta)
            nu = rv.args[1]
            S = rv.args[0]
            code_lines.append(f"    lp += ig_lpdf(para[{i}], {S}, {nu});")
        
        else:
            # Fallback: either raise an exception or insert a comment so you can see
            # where you need to add code for other distributions.
            code_lines.append(f'    // TODO: distribution "{dist_name}" is not yet handled.')
            code_lines.append(f'    // Manually add the log-pdf call for para[{i}].')
    
    code_lines.append("    return lp;")
    code_lines.append("}")
    
    return "\n".join(code_lines)

def generate_dsge_prior_draws(prior_list):
    """
    Generate C++ code for a function that draws from priors
    using Stan Math random number generators.

    Parameters
    ----------
    prior_list : list
        List of SciPy frozen distributions.

    Returns
    -------
    str
        C++ code for generate_prior_draws function.
    """
    code_lines = []
    code_lines.append("template <typename RNG>")
    code_lines.append("std::vector<std::vector<double>> generate_prior_draws(int nsim, RNG& rng) {")
    code_lines.append(f"    std::vector<std::vector<double>> draws(nsim, std::vector<double>({len(prior_list)}));")
    code_lines.append("    using stan::math::normal_rng;")
    code_lines.append("    using stan::math::gamma_rng;")
    code_lines.append("    using stan::math::beta_rng;")
    code_lines.append("    using stan::math::uniform_rng;")
    code_lines.append("    using stan::math::inv_gamma_rng;  // if you write this wrapper")
    code_lines.append("")

    code_lines.append("    for (int i = 0; i < nsim; ++i) {")
    for j, rv in enumerate(prior_list):
        dist_name = rv.dist.name.lower()

        if dist_name == 'norm':
            mu = rv.kwds['loc']
            sigma = rv.kwds['scale']
            code_lines.append(f"        draws[i][{j}] = normal_rng({mu}, {sigma}, rng);")

        elif dist_name == 'gamma':
            shape = rv.args[0]
            scale = rv.kwds.get('scale', 1.0)
            rate = 1.0 / scale
            code_lines.append(f"        draws[i][{j}] = gamma_rng({shape}, {rate}, rng);")

        elif dist_name == 'beta':
            alpha = rv.args[0]
            beta = rv.args[1]
            code_lines.append(f"        draws[i][{j}] = beta_rng({alpha}, {beta}, rng);")

        elif dist_name == 'uniform':
            low = rv.kwds['loc']
            high = low + rv.kwds['scale']
            code_lines.append(f"        draws[i][{j}] = uniform_rng({low}, {high}, rng);")

        elif dist_name == 'invgamma_zellner':
            # Zellner-style inverse gamma with (scale, dof)
            S = rv.args[0]
            nu = rv.args[1]
            code_lines.append(f"        draws[i][{j}] = ig_rng({S}, static_cast<int>({nu}), rng);")

        else:
            code_lines.append(f'        // TODO: Distribution "{dist_name}" not handled.')
            code_lines.append(f'        // draws[i][{j}] = ???;')

    code_lines.append("    }")
    code_lines.append("    return draws;")
    code_lines.append("}")

    return "\n".join(code_lines)


def cpp_code_printer(expr, already_declared=False):
    """
    Converts a sympy expression to C++ code using specialized printing for Eigen matrices.
    
    Args:
        expr: A sympy expression or matrix
        
    Returns:
        C++ code string for the expression or matrix
    """
    from sympy import Matrix, eye, zeros
    from sympy.printing.c import C99CodePrinter
    
    class EigenMatrixPrinter(C99CodePrinter):
        """Custom C/C++ code printer for Eigen matrices and Stan Math expressions"""
        
        def __init__(self):
            super().__init__()
            # Use Stan Math functions for better automatic differentiation support
            self._not_supported = set()
            self.known_functions.update({
                'exp': 'stan::math::exp',
                'log': 'stan::math::log',
                'sqrt': 'stan::math::sqrt',
                'pow': 'stan::math::pow',
                'sin': 'stan::math::sin',
                'cos': 'stan::math::cos',
                'tan': 'stan::math::tan',
                'asin': 'stan::math::asin',
                'acos': 'stan::math::acos',
                'atan': 'stan::math::atan',
                'sinh': 'stan::math::sinh',
                'cosh': 'stan::math::cosh',
                'tanh': 'stan::math::tanh',
                'abs': 'stan::math::fabs',
            })
        
        def _print_Pow(self, expr):
            if expr.exp.is_Number and float(expr.exp).is_integer():
                return super()._print_Pow(expr)
            else:
                return f"stan::math::pow({self._print(expr.base)}, {self._print(expr.exp)})"
        
        def _print_Matrix(self, expr, already_declared=False):
            """Print matrices in C++ Eigen format with comma initializer syntax"""
            rows, cols = expr.shape
            
            # Empty matrices
            if rows == 0 or cols == 0:
                return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>({rows}, {cols})"
            
            # Identity matrices
            # if expr == eye(rows) and rows == cols:
            #     return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity({rows}, {cols})"
            
            # # Zero matrices
            # if expr == zeros(rows, cols):
            #     return f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero({rows}, {cols})"
            
            # Check if sparse (more than 50% zeros)
            lines = []
            mat_name = "result"
            if not already_declared:
                lines.append(f"Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {mat_name}({rows}, {cols});")
            lines.append(f"{mat_name}.setZero();")
            
            for i in range(rows):
                for j in range(cols):
                    if expr[i, j] != 0:
                        lines.append(f"{mat_name}({i}, {j}) = {self._print(expr[i, j])};")
            
            return "\n".join(lines)
    
    printer = EigenMatrixPrinter()
    
    # Handle matrices
    if isinstance(expr, sympy.Matrix):
        return printer._print_Matrix(expr, already_declared)
    
    # For scalars and other expressions
    return printer.doprint(expr)

def smc(model, t0=0):
    """
    Generate C++ code for an FHP model.
    
    Args:
        model: An FHP model object
        t0: Starting time
        
    Returns:
        C++ code for the model
    """
    import sympy
    
    # Get model dimensions for the template
    var_count = len(model['variables'])
    shock_count = len(model['shocks'])
    val_count = len(model['values'])
    obs_count = len(model['observables'])
    para_count = len(model['parameters'])
    eps_count = len(model['innovations'])
    k_val = model['k']
    
    cmodel = model.compile_model()
    npara = len(model.parameters)

    
    # Create substitution dictionary
    cpp_subs = {Parameter(p): Parameter(f"para[{i}]") for i, p in enumerate(model.parameters)}
    
    # Create context for parameter expressions
    context_tuple = [(p, Parameter(p)) for p in model.parameters]

    if 'other_para' in model:
        context_tuple += [(p.name, p) for p in model["other_para"]]
    
    context = dict(context_tuple)
    context["exp"] = sympy.exp
    context["log"] = sympy.log
    
    # Setup for auxiliary parameters 
    to_replace = model['auxiliary_parameters']
    to_replace = list(to_replace.items())
    
    from itertools import permutations
    from sympy import default_sort_key, topological_sort
    
    # Create dependency graph for parameters
    edges = [ 
        (i, j)
        for i, j in permutations(to_replace, 2)
        if type(i[1]) not in [float, int] and i[1].has(j[0])
    ]
    
    # Sort parameters by dependency
    para_func = topological_sort([to_replace, edges], default_sort_key)
    
    # Get system matrices
    system_matrices = model.system_matrices
    
    # Define matrices to write
    to_write = [
        'alpha0_cycle', 'alpha1_cycle', 'beta0_cycle',
        'alphaC_cycle', 'alphaF_cycle', 'alphaB_cycle', 'betaS_cycle',
        'alpha0_trend', 'alpha1_trend', 'betaV_trend',
        'alphaC_trend', 'alphaF_trend', 'alphaB_trend',
        'value_gammaC', 'value_gamma', 'value_Cx', 'value_Cs',
        'P', 'R', 'QQ', 'DD2', 'ZZ', 'HH'
    ]
    
    # Generate C++ code for each matrix
    matrix_assignments = []
    
    # Process each matrix with our improved code printer
    already_declared = {"QQ", "ZZ", "HH"}  # set of names already declared externally
    matrix_assignments = []
    for mat_name, mat in zip(to_write, system_matrices):
        substituted_matrix = (mat.subs(para_func)).subs(cpp_subs)
        print(substituted_matrix)
        cpp_code = cpp_code_printer(substituted_matrix, already_declared=mat_name in already_declared)
        print(cpp_code)
        cpp_code = cpp_code.replace("result", mat_name)
        matrix_assignments.append(f"{cpp_code}")

    # Join all matrix assignment code blocks
    sims_mat = "\n        ".join(matrix_assignments)
    
    log_prior = generate_dsge_logprior(cmodel.prior.priors)

    prior_draws = generate_dsge_prior_draws(cmodel.prior.priors)

    # Generate complete template
    template = cpp_model.format(
        var_count=var_count,
        shock_count=shock_count,
        val_count=val_count,
        obs_count=obs_count,
        para_count=para_count,
        eps_count=eps_count,
        k_val=k_val,
        t0=t0, 
        sims_mat=sims_mat,
        log_prior=log_prior,
        prior_draws=prior_draws
    )
    
    return template

def translate_cpp(model, output_dir="."):
    """
    Translate a model to C++ and write to files.
    
    Args:
        model: The model to translate
        output_dir: Directory to write files to
        
    Returns:
        None
    """
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    except:
        print("Directory already exists.")
    
    # Generate model code
    model_code = smc(model)
    
    # Write model header
    with open(os.path.join(output_dir, "model_t.hpp"), 'w') as f:
        f.write(model_code)
    
    # Compile model to get data
    compiled_model = model.compile_model()
    
    # Write prior file
    from .translate import write_prior_file
    write_prior_file(compiled_model.prior, output_dir)
    
    # Write data file
    np.savetxt(os.path.join(output_dir, "data.txt"), compiled_model.yy)
    
    # Write CMakeLists.txt for building
    cmake_content = """
cmake_minimum_required(VERSION 3.10)
project(dsge_model)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/eherbst/Dropbox/code/math/lib/eigen_3.3.3/")
include_directories("/home/eherbst/Dropbox/code/math")
include_directories("~/Dropbox/code/math/lib/boost_1.69.0/")
include_directories("~/Dropbox/code/math/lib/sundials_4.1.0/include")

# Add executable
add_executable(check_likelihood check_likelihood.cpp)

# Set compiler flags
target_compile_options(check_likelihood PRIVATE -O3)
"""
    
    with open(os.path.join(output_dir, "CMakeLists.txt"), 'w') as f:
        f.write(cmake_content)
    
    param_values = model.p0()  # Python list of floats
    param_string = ", ".join(f"{val:.16f}" for val in param_values)
    parameter_block = f"// Set up test parameters\npara << {param_string};"



    # Write example driver program for testing
    driver_content = """
#include "model_t.hpp"
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/random.hpp>
/**
 * Helper function to print a matrix with formatting
 */
template <typename Derived>
void printMatrix(const std::string& name, const Eigen::MatrixBase<Derived>& matrix,
                int maxRows = 8, int maxCols = 8) {
    std::cout << name << " (" << matrix.rows() << "x" << matrix.cols() << "):" << std::endl;
    
    // Set precision and formatting
    std::cout << std::fixed << std::setprecision(6);
    
    // Determine dimensions to print
    int rows = std::min(maxRows, (int)matrix.rows());
    int cols = std::min(maxCols, (int)matrix.cols());
    
    // Print the matrix
    for (int i = 0; i < rows; i++) {
        std::cout << "  [";
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(12) << matrix(i, j);
            if (j < cols - 1) std::cout << ", ";
        }
        
        // If we're truncating columns, show ellipsis
        if (cols < matrix.cols()) {
            std::cout << ", ...";
        }
        
        std::cout << "]" << std::endl;
    }
    
    // If we're truncating rows, show ellipsis
    if (rows < matrix.rows()) {
        std::cout << "  ..." << std::endl;
    }
    
    std::cout << std::endl;

    // Save the entire matrix to a file named "name.txt"
    std::ofstream outFile(name + ".txt");
    if (!outFile) {
        std::cerr << "Error: Could not open file " << name + ".txt" << " for writing." << std::endl;
        return;
    }
    // Write matrix values row by row, space-separated.
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            outFile << matrix(i, j);
            if (j < matrix.cols() - 1)
                outFile << " ";
        }
        outFile << "\\n";
    }
    outFile.close();
}

/**
 * Example driver for FHP DSGE model
 */
int main() {
    // Create model instance
    Model model;
    
    std::cout << "===== FHP DSGE Model =====" << std::endl;
    std::cout << "Variables: " << model.nvar << std::endl;
    std::cout << "Shocks: " << model.nshock << std::endl;
    std::cout << "Values: " << model.nval << std::endl;
    std::cout << "State dimension: " << model.ns << std::endl;
    std::cout << "Parameters: " << model.npara << std::endl;
    std::cout << std::endl;
    
    // Try to load data
    try {{
        Eigen::MatrixXd data = model.load_data("data.txt");
        std::cout << "Loaded data with dimensions: " << data.rows() << "x" << data.cols() << std::endl;
    }} catch (const std::exception& e) {{
        std::cout << "Warning: " << e.what() << std::endl;
        std::cout << "Continuing without data..." << std::endl;
    }}
    
    // Set up test parameters
    Eigen::VectorXd para(model.npara);
    
    {parameter_block}
    
    // Initialize system matrices
    Eigen::MatrixXd TT(model.ns, model.ns);
    Eigen::MatrixXd RR(model.ns, model.neps);
    Eigen::MatrixXd QQ(model.neps, model.neps);
    Eigen::VectorXd DD(model.nobs);
    Eigen::MatrixXd ZZ(model.nobs, model.ns);
    Eigen::MatrixXd HH(model.nobs, model.nobs);
    
    std::cout << "Calculating system matrices..." << std::endl;
    int error = model.system_matrices(para, TT, RR, QQ, DD, ZZ, HH);
    
    if (error == 0) {{
        std::cout << "System matrices calculation succeeded!" << std::endl << std::endl;
        
        // Print sample of matrices
        printMatrix("TT", TT);
        printMatrix("RR", RR);
        printMatrix("QQ", QQ);
        printMatrix("DD", DD, 10, 1);
        printMatrix("ZZ", ZZ);
        printMatrix("HH", HH);
    }} else {{
        std::cout << "System matrices calculation failed with error code: " << error << std::endl;
        return 1;
    }}
    std::vector<double> para_vector(para.data(), para.data() + para.size());
    std::cout << "Log Prior: " << dsge_logprior(para_vector) << std::endl;
    
// ----- Automatic Differentiation of Log Prior -----
    {
        // Convert parameters to a vector of stan::math::var
        std::vector<stan::math::var> ad_para(para_vector.begin(), para_vector.end());
        // Compute the log prior with autodiff
        stan::math::var lp_var = dsge_logprior(ad_para);
        // Compute gradients with respect to the parameters
        lp_var.grad();
        std::vector<double> autodiff_grad(ad_para.size());
        for (size_t i = 0; i < ad_para.size(); ++i) {
            autodiff_grad[i] = ad_para[i].adj();
        }
        // Print autodiff gradient
        std::cout << "\\nAutodiff Gradient of Log Prior:" << std::endl;
        for (size_t i = 0; i < autodiff_grad.size(); ++i) {
            std::cout << "dLp/dpara[" << i << "] = " << autodiff_grad[i] << std::endl;
        }
        // Reset adjoints for subsequent calculations
        stan::math::set_zero_all_adjoints();
        
        // ----- Numerical Differentiation (Finite Differences) of Log Prior -----
        double h = 1e-8;
        std::vector<double> num_grad(para_vector.size());
        auto f = [&model](const std::vector<double>& p) -> double {
            return dsge_logprior(p);
        };
        for (size_t i = 0; i < para_vector.size(); ++i) {
            std::vector<double> p_plus = para_vector;
            std::vector<double> p_minus = para_vector;
            p_plus[i] += h;
            p_minus[i] -= h;
            double f_plus = f(p_plus);
            double f_minus = f(p_minus);
            num_grad[i] = (f_plus - f_minus) / (2 * h);
        }
        
        // Compare the gradients
        std::cout << "\\nComparing gradients of log prior (Autodiff vs. Numerical):" << std::endl;
        std::cout << "Index\\tAutodiff\\tNumerical\\tDifference" << std::endl;
        for (size_t i = 0; i < para_vector.size(); ++i) {
            double diff = autodiff_grad[i] - num_grad[i];
            std::cout << i << "\\t" << autodiff_grad[i] << "\\t" 
                      << num_grad[i] << "\\t" << diff << std::endl;
        }
    }
    
{
    std::cout << "\\nGenerating prior draws and computing summary statistics..." << std::endl;
    boost::random::mt19937 rng(12345); // Fixed seed for reproducibility
    int nsim = 100000; // Number of draws

    // Generate the draws
    auto draws = generate_prior_draws(nsim, rng);

    int npara = model.npara;
    std::vector<double> mean(npara, 0.0);
    std::vector<double> m2(npara, 0.0);  // For variance computation using Welford's method

    // Compute running mean and M2 (sum of squares of differences from the mean)
    for (int i = 0; i < nsim; ++i) {
        for (int j = 0; j < npara; ++j) {
            double delta = draws[i][j] - mean[j];
            mean[j] += delta / (i + 1);
            m2[j] += delta * (draws[i][j] - mean[j]);
        }
    }

    // Print results
    std::cout << "\\nParameter\\tMean\\tStdDev" << std::endl;
    for (int j = 0; j < npara; ++j) {
        double variance = m2[j] / (nsim - 1);
        double stddev = std::sqrt(variance);
        std::cout << j << "\\t" << mean[j] << "\\t" << stddev << std::endl;
    }
}


    return 0;
}
"""
    
    driver_content = driver_content.replace(
        "{parameter_block}",
        parameter_block
    )

    with open(os.path.join(output_dir, "check_likelihood.cpp"), 'w') as f:
        f.write(driver_content)
    
    print(f"C++ files written to {output_dir}")
