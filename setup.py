import setuptools
import numpy.distutils.core
from numpy.distutils.core import Extension

import os
import sysconfig
from numpy.distutils.system_info import get_info



try:
    blas_opt = get_info('blas_mkl',notfound_action=2)
    lapack_opt = get_info('lapack_mkl',notfound_action=2)

    extra_info = get_info('mkl', 2)

except:
    blas_opt = get_info('blas',notfound_action=2)
    lapack_opt = get_info('lapack',notfound_action=2)

    extra_info = {'libraries': []}


libs = [blas_opt['libraries'][0], lapack_opt['libraries'][0]] + extra_info['libraries']
lib_dirs = [blas_opt['library_dirs'][0], lapack_opt['library_dirs'][0]]



# from numpy.distutils.misc_util import Configuration

# config = Configuration('dsge',parent_package='',top_path=None)

ext1 = Extension(name='dsge.fortran.cholmod',
                 sources=['dsge/fortran/cholmod.f90'],
                 libraries=libs)

ext2 = Extension(name='dsge.fortran.filter',
                 sources=['dsge/fortran/kf_fortran.f90'],
                 libraries=libs)

ext3 = Extension(name='dsge.fortran.gensysw',
                 sources=['dsge/fortran/gensys_wrapper.f90'],
                 libraries=['gensys']+libs,
                 module_dirs=['build/temp.'+sysconfig.get_platform()+'-2.7'])

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

    numpy.distutils.core.setup(
        name = 'dsge',
        version = '0.0.2',
        platforms = 'linux',
        libraries = [('gensys',
                      {'sources': ['dsge/fortran/gensys.f90'],

                   })],
        packages = ['dsge', 'dsge.tests', 'dsge.extension','dsge.fortran'],
        ext_modules = [ext1,ext2,ext3],
        test_suite='nose.collector',
        tests_require=['nose'],
        install_requires=[
            'pandas',
            'slycot',
            'sympy'
         ],

   )
