FFLAGS =  -O3 -xHost -ipo -fPIC -L/mq/home/m1eph00/nlopt/lib -lnlopt -L/mq/home/m1eph00/lib/mkl -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -heap-arrays 
INCLUDE = -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/  -I/opt/intel/include -I/mq/home/m1eph00/nlopt/include
VPATH=.:/mq/home/m1eph00/code/fortran/base

gensys.o : gensys.f90
	ifort $(FFLAGS) $(INCLUDE) -c $< -o $@	

ggensys : gensys.o
	f2py  -c --opt=-O3 -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread  gensys.o gensys_wrapper.f90 --compiler=intel --fcompiler=intelem -m gensys

meyer_gohde : meyer_gohde_interface.f90
	f2py  -c --opt=-O3 -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread  meyer_gohde_interface.f90 --compiler=intel --fcompiler=intelem -m meyer_gohde 


# for debug --debug-capi
