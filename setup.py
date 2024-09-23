import setuptools
import numpy.distutils.core
# def build_ext(config):

#     config_dict = config.todict()
#     config_dict.pop("name")
#     return config_dict


if __name__ == "__main__":

    from numpy.distutils.misc_util import Configuration

    #config_dict = build_ext(Configuration("dsge", parent_package=None, top_path=None))
    numpy.distutils.core.setup(
        name="dsge",
        version="0.1.4",
        platforms="linux",
        packages=["dsge"],
        test_suite="nose.collector",
        tests_require=["nose"],
        package_data={
             "dsge": [
                 "examples/ar1/*",
                 "examples/DGS/*",
                 "examples/edo/*"
                 "examples/nkmp/*",
                 "examples/schorf_phillips_curve/*",
                 "examples/simple-model/*",
                 "examples/sw/*",
                 'schema/*',
                 'linalg/*',
             ]
         },
        requires=["pandas", "scipy", "sympy", "pyyaml", "numba"],
    )
