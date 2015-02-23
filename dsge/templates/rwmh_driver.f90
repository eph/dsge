!!! rwmh_driver.f90 --- 
!! 
!! Description: Driver for the RWMH simulator.
!! 
!! Author: Ed Herbst [edward.p.herbst@frb.gov]
!! Last-Updated: 01/27/15
!! 
program rwmh_driver
  use mkl95_precision, only: wp => dp

  ! Model Selection
  !use {{nkmp,sw,rbc,news,generic_state_space}}
  use {model}

  use mcmc
  
  implicit none

!  include 'mpif.h'

!!$  abstract interface
!!$     function func (x)
!!$         use mkl95_precision, only: wp => dp
!!$         use news
!!$         real(wp) :: func
!!$         real(wp), intent (in) :: x(npara)
!!$     end function func
!!$  end interface

!!$  procedure (func), pointer :: f => null()
!!$  procedure (func), pointer :: flik => null()

  integer :: nsim

  integer :: mpierror, rank, nproc

  real(wp) :: p0(npara), c, p2(2,npara), lik0

  integer :: nblocks

  integer :: irep 

  real(wp) :: proposal_chol(npara,npara)

  real(wp), allocatable :: parasim(:, :), postsim(:)
  integer, allocatable :: acptsim(:)
  character(len=144) :: folder, cirep, arg, blockfile, cnblocks, cnsim, mfile,basedir,pfile,p0file

  integer :: start_time, finish_time, cr, cm, i, j
  real(wp) :: rate

  logical :: direxists


  ! default settings
  nsim      = 100000
  irep      = 1
  nblocks   = 1
  c         = 0.10_wp
  mfile     = varfile
  pfile     = priorfile
  blockfile = ''
  basedir   = '/mq/scratch/m1eph00/'
  p0file    = 'pmsv'


  proposal_chol = 0.0_wp
  do i = 1, npara
     proposal_chol(i,i) = 1.0_wp
  end do


  call omp_set_num_threads(1)
  call mkl_set_num_threads(1)

  do i = 1, command_argument_count()

     call get_command_argument(i, arg)

     select case(arg)

     case('-n','--nsim')
        call get_command_argument(i+1,arg)
        read(arg, '(i1000)') nsim

     case('-i','--initrep')
        call get_command_argument(i+1, arg)
        read(arg, '(i)') irep

     case('-v','--varfile')
        !        call get_command_argument(i+1, varfile)

     case('-o','--nblocks')
        call get_command_argument(i+1,arg)

        if (arg == 'random') then
           nblocks = -1
        elseif (arg == 'fixed') then
           nblocks = -2
        elseif (arg == 'dist') then
           nblocks = -4
        else
           read(arg, '(i)') nblocks
        end if

     case('-f','--blockfile')
        call get_command_argument(i+1,blockfile)

     case('-c','--scaling')
        call get_command_argument(i+1,arg)
        read(arg,'(f)') c

     case('--output-dir')
        call get_command_argument(i+1, arg)
        basedir = arg
     case('-M')
        call get_command_argument(i+1,mfile)
        open(1, file=mfile, status='old',action='read')
        do j = 1, npara
           read(1,*) proposal_chol(j,:)
        end do
        close(1)
     case('--prior-file')
        call get_command_argument(i+1,pfile)
     case('--p0')
        call get_command_argument(i+1,p0file)
     case('--randinit')

     end select

  end do
  print*,pfile
  open(1, file=pfile, status='old', action='read')
  do i = 1, npara
     read(1, *) pshape(i), pmean(i), pstdd(i), pmask(i), pfix(i)
  end do
  close(1)

  open(1, file=datafile, status='old', action='read')
  do i = 1, nobs
     read(1, *) YY(:,i)
  end do
  close(1)

  open(1, file=transfile, status='old', action='read')
  do i = 1, npara
     read(1, *) trspec(:,i)
  end do
  close(1)



!  call mpi_init(mpierror)
!  call mpi_comm_size(MPI_COMM_WORLD, nproc, mpierror)
!  call mpi_comm_rank(MPI_COMM_WORLD, rank, mpierror)

  irep = irep + rank 
  write(cirep, '(i2)') irep
  write(cnsim,'(i20)') nsim
  if (nblocks > 0) then 
     write(cnblocks,'(i)') nblocks
  elseif (nblocks == -1) then
     cnblocks = 'random'
  elseif (nblocks == -2) then
     cnblocks = 'fixed'
  elseif (nblocks == -4) then
     cnblocks = 'dist'
  end if

  !------------------------------
  ! Timing
  !------------------------------
  call system_clock(count_rate=cr)
  call system_clock(count_max=cm)
  rate = real(cr)

  !-------------------------------
  ! Get objfun
  !-------------------------------
  ! if (any_missing_data(YY)) then
  !    print*,'missing data---using kalman filter'
  !    f => objfunkf_missing
  !    flik => lik_missing
  ! else
  !    f => objfun
  !    flik => lik
  ! end if

  !------------------------------------------------------------
  !
  ! Initialization
  ! 
  !------------------------------------------------------------
  select case(p0file)

  case('random','rand')
     ! Draw from the prior.
     lik0 = -100000000.0_wp
     do while ((lik0 < -100000.0_wp) .and. (i < 1000))
        p2 = priorrand(2, pshape, pmean, pstdd, pmask, pfix)
        p0 = p2(1,:)
        lik0 = objfun(p0)
        if (isnan(lik0)) lik0 = -10000000.0_wp; 

        i = i + 1;
     end do

     if (i == 1000) then
        print*,'Could not find a valid starting draw...aborting.'
        stop
     end if


  case('pmsv')
     ! draw model p0
     p0 = pmsv()
  case default
     ! start from file
     open(1,file=p0file,status='old',action='read')
     do i = 1,npara
        read(1,*) p0(i)
     end do
  end select
        
  lik0 = objfun(p0)
  print*,lik0!,objfun(p0)

  ! Use a prespecificed starting location.
 folder = 'rwmh-'//mname//'-nsim-'//trim(adjustl(cnsim))//'-nblocks-'//&
       trim(adjustl(cnblocks))//'-trial'//trim(adjustl(cirep))

  print*, 'Saving files to:', folder
  print*, ''
  print*, 'In directory:', trim(adjustl(basedir))

  inquire(directory=trim(adjustl(basedir))//trim(adjustl(folder)), exist=direxists)
  if (direxists) then
     !print *, 'directory already exists'
  else
     call system('mkdir '//trim(adjustl(basedir))//trim(adjustl(folder)))
  endif

  open(1,file=trim(adjustl(basedir))//trim(adjustl(folder))//'/options.txt', action='write')
  write(1,'(A20,A)')  'model     : ', mname
  write(1,'(A20,i)')  'nsim      : ', nsim
  write(1,'(A20,A)')  'nblocks   : ', cnblocks
  write(1,'(A20,f)')  'c         : ', c
  write(1,'(A20,A)')  'varfile   : ', trim(adjustl(mfile))
  write(1,'(A20,A)')  'priofile  : ', trim(adjustl(pfile))
  write(1,'(A20,A)')  'p0        : ', trim(adjustl(p0file))
  close(1)

  folder = trim(adjustl(basedir))//trim(adjustl(folder))

  call system_clock(start_time)
  allocate(parasim(npara, nsim), postsim(nsim),acptsim(nsim))
  call rwmh(p0, c**2*proposal_chol, nsim, objfun, npara, pmask, parasim, postsim, acptsim, nblocks,blockfile,folder)
  deallocate(parasim, postsim,acptsim)
  call system_clock(finish_time)
  write(*,'(a,f7.3,a)') "Elapsed time is ", (finish_time-start_time)/rate, " seconds."

!  call mpi_barrier(MPI_COMM_WORLD, mpierror)
!  call mpi_finalize(mpierror)

end program rwmh_driver
