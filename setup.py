from setuptools import setup, find_packages
# def build_ext(config):

#     config_dict = config.todict()
#     config_dict.pop("name")
#     return config_dict


if __name__ == "__main__":
    # Legacy fallback for environments still invoking setup.py directly.
    # Primary build is defined in pyproject.toml (PEP 517/621 with setuptools).
    setup(
        name="dsge",
        platforms="linux",
        packages=find_packages(),  # Use find_packages instead of ["dsge"]
        install_requires=[
            "pandas",
            "scipy",
            "sympy",
            "pyyaml",
            "numba",
            "lark",
        ],
        include_package_data=True,
        package_data={
            "dsge": [
                "examples/ar1/*",
                "examples/DGS/*",
                "examples/edo/*",
                "examples/nkmp/*",
                "examples/schorf_phillips_curve/*",
                "examples/simple-model/*",
                "examples/sw/*",
                "schema/*",
                "linalg/*",
                "*.py",
            ]
        },
    )
