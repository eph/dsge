#!/usr/bin/env python3
"""
Development installation script for the DSGE package.

This script helps with installing the DSGE package in development mode,
ensuring that all modules are available for testing.
"""

import os
import subprocess
import sys

def install_dev():
    """Install the DSGE package in development mode."""
    print("Installing DSGE package in development mode...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Install the package in development mode
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', script_dir])
    
    print("\nDSGE package installed in development mode.")
    print("You can now run tests or import from dsge in your Python code.")

if __name__ == "__main__":
    install_dev()