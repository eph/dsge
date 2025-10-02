#!/home/eherbst/miniconda3/bin/python
"""
Display FHP DSGE model Fortran code.
"""

import os
import sys

# Add the current directory to the path so we can import dsge
sys.path.insert(0, os.path.abspath('.'))

# Import required modules 
from dsge.parse_yaml import read_yaml
from dsge.translate import make_fortran_model

def main():
    """
    Main function to demonstrate FHP model translation to Fortran.
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
    output_dir = '_fortress_tmp_fortran'
    
    # Translate model to Fortran
    print(f"Translating model to Fortran in {output_dir}...")
    make_fortran_model(model, output_directory=output_dir)
    
    print(f"\nTranslated FHP model to Fortran code in directory: {output_dir}")
    
if __name__ == "__main__":
    main()