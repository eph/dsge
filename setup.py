from setuptools import find_packages
import numpy.distutils.core

from setuptools import find_packages

def build_ext(config):

    # from numpy.distutils.system_info import get_info

    # lapack_info = get_info('lapack_opt',1)

    # f_sources = ['dsge/fortran/cholmod.f90',
    #              'dsge/fortran/kf_fortran.f90',
    #              'dsge/fortran/gensys_wrapper.f90']

    

    # #lapack_info['libraries'].remove('mkl_lapack95_lp64')

    # if lapack_info:
    #     config.add_library('gensys', 'dsge/fortran/gensys.f90')


    #     config.add_extension(name='fortran.cholmod',
    #                          sources=['dsge/fortran/cholmod.f90'],
    #                          extra_info = lapack_info)##libraries=['lapack', 'blas'])

    #     config.add_extension(name='fortran.filter',
    #                          sources=['dsge/fortran/kf_fortran.f90'],
    #                          extra_info = lapack_info)#                             libr


    #     config.add_extension(name='fortran.gensysw',
    #                          sources=['dsge/fortran/gensys.f90'],
    #                          libraries = ['gensys'],
    #                          extra_info = lapack_info,
    #                          )

    config_dict = config.todict()
    config_dict.pop('name')
    return config_dict
#config_dict.pop('packages')


# from numpy.distutils.misc_util import Configuration

# config = Configuration('dsge',parent_package='',top_path=None)

# ext1 = Extension(name='dsge.fortran.cholmod',
#                  sources=['dsge/fortran/cholmod.f90'],
#                  libraries=libs)

# ext2 = Extension(name='dsge.fortran.filter',
#                  sources=['dsge/fortran/kf_fortran.f90'],
#                  libraries=libs)

# ext3 = Extension(name='dsge.fortran.gensysw',
#                  sources=['dsge/fortran/gensys_wrapper.f90'],
#                  libraries=['gensys']+libs,
#                  module_dirs=['build/temp.'+sysconfig.get_platform()+'-2.7'])

# config.version = '0.0.2'

#config.add_library('gensys', 'dsge/fortran/gensys.f90')

# ext3 = Extension() #libraries = ['slicot'] + libs,
#                  #library_dirs = ['/mq/home/m1eph00/lib'])
# # ext4 = Extension(name = 'dsge.fortran.dlyap',
# #                  sources = ['dsge/fortran/dlyap_wrapper.f90'],
# #                  libraries = ['slicot'] + libs,
# #                  library_dirs = ['/mq/home/m1eph00/lib'],
# #                  f2py_options = ['--verbose'])




if __name__ == "__main__":

    from numpy.distutils.misc_util import Configuration
    config_dict = build_ext(Configuration('dsge',parent_package=None,
                                          top_path=None))
    numpy.distutils.core.setup(
        name = 'dsge',
        version = '0.0.2',
        platforms = 'linux',
        packages = find_packages(),
        test_suite='nose.collector',
        tests_require=['nose'],
        package_data = {'dsge':
                        ['examples/ar1/*',
                         'examples/DGS/*',
                         'examples/edo/*',
                         'examples/nkmp/*',
                         'examples/schorf_phillips_curve/*',
                         'examples/simple-model/*',
                         'examples/sw/*']},
        install_requires=[
            'pandas',
            #'slycot',
            #'sympy',
            #'scipy'
         ],
   )
