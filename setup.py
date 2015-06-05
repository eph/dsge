import setuptools
import numpy.distutils.core
from numpy.distutils.core import Extension

import os

from numpy.distutils.system_info import get_info

try:
    blas_opt = get_info('blas_mkl',notfound_action=2)
    lapack_opt = get_info('lapack_mkl',notfound_action=2)

    extra_info = get_info('mkl', 2)

except:
    blas_opt = get_info('blas',notfound_action=2)
    lapack_opt = get_info('lapack',notfound_action=2)

    
libs = [blas_opt['libraries'][0], lapack_opt['libraries'][0]] + extra_info['libraries']
lib_dirs = [blas_opt['library_dirs'][0], lapack_opt['library_dirs'][0]]

ext1 = Extension(name = 'dsge.fortran.cholmod',
                 sources = ['dsge/fortran/cholmod.f90'],
                 libraries = libs)
ext2 = Extension(name = 'dsge.fortran.gensysw',
                 sources = ['dsge/fortran/gensys_wrapper.f90'],
                 libraries = ['gensys'] + libs, 
                 module_dirs = ['/mq/home/m1eph00/python-repo/dsge/build/temp.linux-x86_64-2.7/']
                 )
ext3 = Extension(name = 'dsge.fortran.filter',
                 sources = ['dsge/fortran/kf_fortran.f90'],
                 libraries = ['slicot'] + libs,
                 library_dirs = ['/mq/home/m1eph00/lib'],
                 )
ext4 = Extension(name = 'dsge.fortran.dlyap',
                 sources = ['dsge/fortran/dlyap_wrapper.f90'],
                 libraries = ['slicot'] + libs,
                 library_dirs = ['/mq/home/m1eph00/lib'], 
                 f2py_options = ['--verbose'])


if __name__ == "__main__":

    numpy.distutils.core.setup(

        name = 'dsge',
        version = '0.0.2',
        platforms = 'linux',
        libraries = [('gensys',
                      {'sources': ['dsge/fortran/gensys.f90'],
                   })],
        packages = ['dsge', 'dsge.tests'],
        ext_modules = [ext1,ext2,ext3,ext4],
        test_suite='nose.collector',
        tests_require=['nose']
    )
