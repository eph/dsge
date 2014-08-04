F90 = ifort
FC = ifort

MPIFC = /usr/bin/mpif90 -f90=ifort
SPEED = -O3 -xHost -ipo
FFLAGS =  -O3 -xHost -ipo -fPIC -L/mq/home/m1eph00/nlopt/lib -lnlopt -L/mq/home/m1eph00/lib/mkl -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -lpthread -heap-arrays 
INCLUDE = -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/  -I/opt/intel/include -I/mq/home/m1eph00/nlopt/include
LIBPATH = -L/mq/home/m1eph00/lib
LIB = -lslicot_sequential

VPATH = .:base:base/amiller:model:
F90FLAGS = $(FFLAGS) -L/mq/home/m1eph00/lib -lslicot_sequential -I/opt/intel/mkl/include -I/opt/intel/mkl/include/em64t/lp64  -I/opt/intel/include
OBJDIR = obj



# drivers 
smc_driver_mpi : prior.o smc_driver_mpi.f90 rand.o {model}.o mcmc.o
	$(MPIFC) $(FFLAGS) $(INCLUDE) -o smc_{model} rand.o hessian.o mcmc.o as63.o prior.o {model}.o smc_driver_mpi.f90 $(LIBPATH) $(LIB)

rwmh_driver : prior.o rwmh_driver.f90 rand.o {model}.o mcmc.o hessian.o
	$(MPIFC) $(FFLAGS) $(INCLUDE) -o rwmh_{model} rand.o hessian.o mcmc.o as63.o prior.o {model}.o rwmh_driver.f90 $(LIBPATH) $(LIB) -check bounds


#------------------------------------------------------------ 
# models
#------------------------------------------------------------
# dssw_jp.o : dssw_jp.f90 prior.o filter.o gensys.o helper_functions.f90 prior.f90
# 	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@	
{model}.o : {model}.f90 prior.o meyer_gohde.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

# prior -- requires as63
prior.o : prior.f90 as63.o
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

# stand-alone mod files
hessian.o : hessian.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

mcmc.o : mcmc.f90 rand.o hessian.o
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

as63.o: as63.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

filter.o: filter.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@

gensys.o : gensys.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@	

strid_walentin.o : strid_walentin.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@		

rand.o : rand.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c $< -o $@	

clean : 
	rm -f *.mod *.o



