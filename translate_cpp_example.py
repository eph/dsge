#!/usr/bin/env python3
"""
Example script demonstrating how to translate an FHP model to C++ code.
"""

import os
import sys

# Add the current directory to the path so we can import dsge
sys.path.insert(0, os.path.abspath('.'))

from dsge.parse_yaml import read_yaml
from dsge.translate_cpp import translate_cpp

def main():
    """
    Main function to demonstrate FHP model translation to C++.
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
    output_dir = '_fortress_tmp_cpp'
    
    # Translate model to C++
    print(f"Translating model to C++ in {output_dir}...")
    translate_cpp(model, output_dir=output_dir)
    
    print(f"\nTranslated FHP model to C++ code in directory: {output_dir}")
    print("The following files were created:")
    print(f"  - {output_dir}/model_t.hpp - C++ model implementation using Eigen and stan-math")
    print(f"  - {output_dir}/check_likelihood.cpp - Example driver program")
    print(f"  - {output_dir}/CMakeLists.txt - CMake build configuration")
    print(f"  - {output_dir}/data.txt - Data file")
    print(f"  - {output_dir}/prior.txt - Prior specification file")
    print("\nTo build and run:")
    print(f"1. Edit {output_dir}/CMakeLists.txt to set your Stan Math path")
    print(f"2. Navigate to {output_dir}")
    print("3. Run: mkdir build && cd build")
    print("4. Run: cmake .. && make")
    print("5. Run: ./check_likelihood")

if __name__ == "__main__":
    main()
