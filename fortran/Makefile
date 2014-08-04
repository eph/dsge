
ifeq ($(user),m1eph00)
	FFLAGS =  -O3 -xHost -ipo -fPIC -L/mq/home/m1eph00/nlopt/lib -lnlopt -L/mq/home/m1eph00/lib/mkl -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -heap-arrays
	INCLUDE = -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/  -I/opt/intel/include -I/mq/home/m1eph00/nlopt/include
	VPATH=.:/mq/home/m1eph00/code/fortran/base
	FC=ifort

	F2PYFLAGS=-I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread   --compiler=intel --fcompiler=intelem

	SLICOT=-lslicot_sequential
else
	FFLAGS=-llapack -lblas
	INCLUDE=
	VPATH=.
	FC=gfortran
	F2PYFLAGS=$(FFLAGS)
	SLICOT=-lslicot
endif
gensys.o : gensys.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

gensys : gensys.o
	f2py  -c --opt=-O3 $(F2PYFLAGS) gensys.o gensys_wrapper.f90  -m gensys

meyer_gohde : meyer_gohde_interface.f90
	f2py  -c --opt=-O3 $(F2PYFLAGS) meyer_gohde_interface.f90 -m meyer_gohde

dlyap : dlyap_wrapper.f90
	f2py  -c --opt=-O3 $(F2PYFLAGS) $(SLICOT) dlyap_wrapper.f90 -m dlyap
# for debug --debug-capi
