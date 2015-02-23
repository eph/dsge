import setuptools
import numpy.distutils.core
from numpy.distutils.core import Extension

from numpy.distutils.system_info import get_info

blas_opt = get_info('blas',notfound_action=2)
lapack_opt = get_info('lapack',notfound_action=2)

libs = [blas_opt['libraries'][0], lapack_opt['libraries'][0]]
lib_dirs = [blas_opt['library_dirs'][0], lapack_opt['library_dirs'][0]]

print libs
ext1 = Extension(name = 'dsge.fortran.cholmod',
                 sources = ['dsge/fortran/cholmod.f90'])
ext2 = Extension(name = 'dsge.fortran.gensysw',
                 sources = ['dsge/fortran/gensys_wrapper.f90'],
                 libraries = ['gensys','blas', 'lapack'],
                 module_dirs = ['/home/eherbst/from-work/dsge/build/temp.linux-i686-2.7/']
                 )
ext3 = Extension(name = 'dsge.fortran.filter',
                 sources = ['dsge/fortran/kf_fortran.f90'],
                 libraries = ['slicot','blas', 'lapack'],
                 )
ext4 = Extension(name = 'dsge.fortran.dlyap',
                 sources = ['dsge/fortran/dlyap_wrapper.f90'],
                 libraries = ['slicot','blas','lapack'],
                 f2py_options = ['--verbose'])


if __name__ == "__main__":

    numpy.distutils.core.setup(

        name = 'dsge',
        version = '0.0.1',
        platforms = 'linux',
        libraries = [('gensys',
                      {'sources': ['dsge/fortran/gensys.f90'],
                   })],
        packages = ['dsge'],
        ext_modules = [ext1,ext2,ext3,ext4],

        test_suite='nose.collector',
        tests_require=['nose']
    )
