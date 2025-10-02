#!/usr/bin/env python3
"""
Export FHP DSGE model to C++ with improved code generation.

This script demonstrates the improved C++ code generation feature
that uses Eigen matrices and stan-math for automatic differentiation.
"""

import os
import sys

# Add the current directory to the path so we can import dsge
sys.path.insert(0, os.path.abspath('.'))

# Import required modules
from dsge.parse_yaml import read_yaml
from dsge.translate_cpp import translate_cpp

def main():
    """
    Export an FHP model to C++ code.
    """
    # Path to FHP model YAML file
    yaml_path = os.path.join('dsge', 'examples', 'fhp', 'fhp.yaml')
    
    # Check if file exists
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        print("Make sure you're running this from the project root directory.")
        return
    
    print(f"Loading FHP model from {yaml_path}...")
    
    # Read YAML file
    model = read_yaml(yaml_path)
    
    # Create output directory
    output_dir = '_improved_cpp_export'
    
    # Translate model to C++
    print(f"Translating model to C++ with improved code generation in {output_dir}...")
    translate_cpp(model, output_dir=output_dir)
    
    print(f"\nTranslated FHP model to C++ code with advanced matrix syntax in directory: {output_dir}")
    print("The following files were created:")
    print(f"  - {output_dir}/model_t.hpp - C++ model implementation using Eigen and stan-math")
    print(f"  - {output_dir}/check_likelihood.cpp - Enhanced driver program with matrix visualization")
    print(f"  - {output_dir}/CMakeLists.txt - CMake build configuration")
    print(f"  - {output_dir}/data.txt - Data file")
    print(f"  - {output_dir}/prior.txt - Prior specification file")
    
    print("\nFeatures of the improved code generation:")
    print("  - Uses optimized Eigen matrix constructors (Identity, Zero) for special matrices")
    print("  - Detects sparse matrices and uses element-wise assignment for better readability")
    print("  - Uses dense comma initializer syntax for compact matrix definitions")
    print("  - Proper integration with Stan Math library for automatic differentiation")
    print("  - Enhanced visualization with formatted matrix output")
    
    print("\nTo build and run:")
    print(f"1. Edit {output_dir}/CMakeLists.txt to set your Stan Math path")
    print(f"2. Navigate to {output_dir}")
    print("3. Run: mkdir build && cd build")
    print("4. Run: cmake .. && make")
    print("5. Run: ./check_likelihood")

if __name__ == "__main__":
    main()
