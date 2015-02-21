!!! blockmcmc.f90 --- 
!! 
!! Description: Driver for the block MH sampler.
!! 
!! Author: Ed Herbst [edward.p.herbst@frb.gov]
!! Last-Updated: 01/27/15
!! 
program blockmcmc

  use mkl95_precision, only: wp => dp

  use mcmc

  ! Model Selection
  use {model}

  implicit none

  !------------------------------------------------------------
  ! Variable Definitions
  !------------------------------------------------------------
  integer :: nsim 
  real(wp), allocatable :: parasim(:, :), postsim(:)
  integer, allocatable  :: acptsim(:, :)

  ! blocking parameters
  integer  :: nblocks, ind(npara), break_points(npara+1)
  integer  :: max_block_size, block_burn
  real(wp) :: bprob

  real(wp) :: p0(npara), f0

  character(len=144) :: arg, mfile, blockfile, mumethod, fstr
  character(len=144) :: cnsim, cnblocks, cirep 

  logical :: nest, direxists, tran
  real(wp) :: sa_options(5), sbar, M(npara, npara), c
  integer :: method, methodu, status, seed

  ! timing
  integer :: start_time, finish_time, cr, cm, df, irep
  real(wp) :: rate

  integer :: i, j

  !------------------------------------------------------------
  !
  ! Read in prior, data, transform, prespecified-variance
  !
  !------------------------------------------------------------
  open(1, file=priorfile, status='old', action='read')
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
!  open(2, file=varfile, status='old', action='read')
  M = 0.0_wp
  do i = 1, npara
     read(1, *) trspec(:,i)
!     read(2, *) M(i,:)
  end do
  close(1)
!  close(2)



  !------------------------------------------------------------
  !
  ! Default values for the algorithm.
  !
  !------------------------------------------------------------
  nsim           = 10000        

  !------------------------------
  ! Proposal Centering/Optimization 
  !------------------------------
  mumethod       = 'infomat'    ! optimization routine
  sbar           = 2.0_wp       ! upper bound of step size

  ! proposal variance
  df             = 10           ! df of Student-t Distribution
  c              = 1.0_wp       ! scaling of proposal

  tran           = .false.      ! used transformed parameters

  ! Simulated Annealing Options -- See CR 2010
  sa_options     = (/4.0_wp, 5.0_wp, 6.0_wp, 0.4_wp, 1.0_wp/) 

  !------------------------------
  ! Block options
  !------------------------------
  ! random blocks, fixed size
  nblocks        = 1         
   
  ! randomly sized blocks
  bprob          = 0.85_wp      ! Probability of new block

  ! distributional blocks
  blockfile      = ''           ! Name of blockfile
  block_burn     = 200          ! Stop after burn-in period.

  ! Hessian-based blocking
  max_block_size = 2            ! Maximum blocksize

  irep           = 1            ! Monte Carlo index.

  

  p0 = pmsv()

  fstr = '/mq/scratch/m1eph00/'
  !------------------------------
  ! Threading Options
  !------------------------------
  call omp_set_num_threads(1)
  call mkl_set_num_threads(6)
  call omp_set_dynamic(.true.)


  !------------------------------------------------------------
  !
  ! Read in command line options. 
  !
  !------------------------------------------------------------
  do i = 1, command_argument_count()

     call get_command_argument(i, arg)

     select case(arg)
     case('--help')
        call print_help()
        stop
     case('-i', '--initrep')
        call get_command_argument(i+1, arg)
        read(arg, '(i)') irep

     case('-n','--nsim')
        call get_command_argument(i+1,arg)
        read(arg, '(i)') nsim
      
     case('-c','--scale')
        call get_command_argument(i+1,arg)
        read(arg, '(f)') c

     case('--df')
        call get_command_argument(i+1,arg)
        read(arg, '(i)') df

     case('-o','--nblocks')
        call get_command_argument(i+1,arg)

        if (arg == 'random') then
           nblocks = -1
        elseif (arg == 'fixed') then
           nblocks = -2
        elseif (arg == 'hessian') then
           nblocks = -3
        elseif (arg == 'dist') then
           nblocks = -4
        else
           read(arg, '(i)') nblocks
        end if

     case('--newton')
        mumethod = 'newton'
        call omp_set_num_threads(6)
        call mkl_set_num_threads(2)
        call omp_set_dynamic(.true.)

     case('--infomat')
        mumethod = 'infomat'
        call omp_set_num_threads(1)
        call mkl_set_num_threads(6)
        call omp_set_dynamic(.true.)

     case('--sa')
        mumethod = 'sa'
        call omp_set_num_threads(6)
        call mkl_set_num_threads(2)

     case('--sa-opt') 
        call get_command_argument(i+1,arg)
        read(arg, '(f)') sa_options(1)

        call get_command_argument(i+2,arg)
        read(arg, '(f)') sa_options(2)

        call get_command_argument(i+3,arg)
        read(arg, '(f)') sa_options(3)

        call get_command_argument(i+4,arg)
        read(arg, '(f)') sa_options(4)

        call get_command_argument(i+5,arg)
        read(arg, '(f)') sa_options(5)

     case('-s','--sbar')
        call get_command_argument(i+1,arg)
        read(arg, '(f)') sbar
     case('--output-dir')
        call get_command_argument(i+1,fstr)

     case('--langevin')
        mumethod = 'langevin'
        call omp_set_num_threads(6)
        call mkl_set_num_threads(2)

        call omp_set_dynamic(.true.)
     case('--newton-sameR')
        mumethod = 'newton-sameR'

        call omp_set_num_threads(12)
        call mkl_set_num_threads(1)

        call omp_set_dynamic(.true.)
     case('--infomat-sameR')
        mumethod = 'infomat-sameR'
     case('--riemann')
        mumethod = 'riemann'

     case('--block-burn')
        call get_command_argument(i+1,arg)
        read(arg, '(i)') block_burn

     case('--max-block')
        call get_command_argument(i+1,arg)
        read(arg, '(i)') max_block_size
     case('-f','--blockfile')
        call get_command_argument(i+1,blockfile)

     case('-M')
        call get_command_argument(i+1,mfile)
        open(1, file=mfile, status='old',action='read')
        do j = 1, npara
           read(1,*) M(j,:)
        end do
        close(1)
     case('--p0')
        call get_command_argument(i+1,mfile)
        open(1, file=mfile, status='old',action='read')
        do j = 1, npara
           read(1,*) p0(j)
        end do
        close(1)
     end select
     

  end do


  write(cnsim,'(i)') nsim
  if (nblocks > 0) then 
     write(cnblocks,'(i)') nblocks
  elseif (nblocks == -1) then
     cnblocks = 'random'
  elseif (nblocks == -2) then
     cnblocks = 'fixed'
  elseif (nblocks == -4) then
     cnblocks = 'dist'
  else 
     cnblocks = 'hessian'
  end if
  write(cirep,'(i)') irep

  fstr = trim(adjustl(fstr))//'mc-'//trim(adjustl(mname)) &
       //'-'//trim(adjustl(mumethod))//'-nsim-' &
       //trim(adjustl(cnsim))//'-nblocks-'//trim(adjustl(cnblocks)) &
       //'-trial-'//trim(adjustl(cirep))//'/'

  print*, 'Saving files to:', fstr
  inquire(file=fstr, exist=direxists)
  if (direxists) then
     print *, 'directory already exists'
  else
     call system('mkdir '//fstr)
  endif
  ! write options file
  open(1,file=trim(adjustl(fstr))//'options.txt',action='write')
  write(1,'(A20,A)') 'model          : ', mname
  write(1,'(A20,i)') 'nsim           : ', nsim
  write(1,'(A20,A)') 'method         : ', mumethod
  write(1,'(A20,A)') 'nblocks        : ', cnblocks
  write(1,'(A20,i)') 'df             : ', df
  write(1,'(A20,f)') 'sbar           : ', sbar
  write(1,'(A20,f)') 'c              : ', c
  write(1,'(A20,f)') 'prob include   : ', bprob
  write(1,'(A20,i)') 'block burn     : ', block_burn
  write(1,'(A20,i)') 'max block size : ', max_block_size
  close(1)


  allocate(parasim(npara,nsim),postsim(nsim),acptsim(npara,nsim))  



  print*,p0
  print*,objfun(p0)
  call blockmh(p0, M, nsim, objfun, test_analytic_derivative, inbounds, npara, nblocks, pmask, & 
       parasim, postsim, acptsim, mumethod, sa_options, df,sbar,c,bprob,block_burn, & 
       max_block_size, blockfile,trim(adjustl(fstr)))

  open(1,file=trim(adjustl(fstr))//'parasim.txt',action='write')
  do i = 1, nsim
     write(1,'(100f)') parasim(:,i)
  end do
  close(1)

  open(1,file=trim(adjustl(fstr))//'postsim.txt',action='write')
  do i = 1, nsim
     write(1,'(100f)') postsim(i)
  end do
  close(1)
  print*,'finished writing files to ...'
  print*,trim(adjustl(fstr))//'parasim.txt'
  deallocate(parasim,postsim,acptsim)



  contains 
    subroutine test_analytic_derivative(para, psel, neffpara, loglh, dloglh, info_mat)

      integer, intent(in) :: neffpara, psel(neffpara)
      real(wp), intent(inout) :: para(npara)

      real(wp), intent(out) :: loglh, dloglh(neffpara), info_mat(neffpara,neffpara)
      real(wp) :: hlnprior(npara,npara), dlnprior(npara)
      integer info


      real(wp) :: TT(ns,ns), RR(ns,neps), QQ(neps,neps), DD(ny), ZZ(ny,ns), HH(ny,ny)
      real(wp) :: DTT(ns**2,neffpara), DRR(ns*neps,neffpara), DQQ(neps**2,neffpara)
      real(wp) :: DDD(ny,neffpara), DZZ(ny*ns,neffpara), DHH(ny**2,neffpara)

      if (inbounds(para) .ne. .true.) then
         loglh = REALLY_NEG

         dloglh = 0.0_wp
         info_mat = 0.0_wp
         do info = 1,neffpara
            info_mat(info,info)= 10000.0_wp
         end do

         return
      end if

      call sysmat(para,TT,RR,QQ,DD,ZZ,HH,info)
!      call sysmat_diff(para,psel,neffpara,DTT,DRR,DQQ,DDD,DZZ,DHH)

      call chand_recursion_derivative(YY, TT, RR, QQ, DD, ZZ, HH, DTT, DRR, DQQ, DDD, DZZ, DHH, & 
           ny, nobs, neps, ns, t0, neffpara, loglh, dloglh, info_mat)    

      dlnprior = dlpriordens(para, pmean, pstdd, pshape, pmask, pfix)
      hlnprior = hlpriordens(para, pmean, pstdd, pshape, pmask, pfix)

      loglh = loglh + priordens(para, pmean, pstdd, pshape, pmask, pfix)
      dloglh   = dloglh + dlnprior(psel)
      info_mat = info_mat - hlnprior(psel,psel)

    end subroutine test_analytic_derivative

    subroutine print_help()
      print '(a)', 'blockmcmc -- Block Metropolis-Hastings Algorithms for DSGE models'
      print '(a)', '      by Ed Herbst [edward.p.herbst@frb.gov]'
      print '(a)', ''
      print '(16a,a)', 'current model  : ', mname
      print '(a)', ''
      print '(a)', 'usage: ./blockmcmc [OPTIONS]'
      print '(a)', ''
      print '(a)', 'options:'
      print '(a)', ''
      print '(a)', '-n, --nsim [N]          sets nphi = N                       DEFAULT = 10000'
      print '(a)', '-i, --initrep [N]       sets the trial number = N           DEFAULT = 1'
      print *,''
      print *,''

      print '(a)', 'Proposal Options:'
      print '(a)', '--newton                uses newtonion opt. for proposal'
      print '(a)', '--newton-sameR          uses PSP-MALA for proposal'
      print '(a)', '--infomat               uses a (pseudo) information matrix for proposal'
      print '(a)', '--infomat-sameR         uses PSP-MALA with (pseudo) info. mat. for proposal'
      print '(a)', '--riemann               uses riemann manifold langevin with (pseudo) infomat'
      print '(a)', '--langevin              uses P-MALA for proposal'
      print '(a)', '--sa                    uses simulated annealing'
      print '(a)', '--sa-opt [N T I R C]    sets simulated annealing options'
      print '(a)', '      N: number of stages'
      print '(a)', '      T: initial temperature'
      print '(a)', '      I: stage length increment'
      print '(a)', '      R: temperature reduction factor'
      print '(a)', '      C: initial scaling'

      print '(a)', '-c --scale [c]          scales the proposal variance by c   DEFAULT = 1'
      print '(a)', '--df [N]                sets the proposal degree of freedom DEFAULT = 10'
      print '(a)', '-M [FILENAME]           sets proposal variance from FILENAME for P-MALA' 
      print*,''
      print*,''
      print '(a)', 'Blocking options:'
      print '(a)', '-o, --nblocks [N]       sets number of blocks = N           DEFAULT = 1'
      print '(a)', '-o, --nblocks random    sets number random number of blocks' 
      print '(a)', '-o, --nblocks fixed     sets a fixed number of blocks from blockfile'
      print '(a)', '-o, --nblocks dist      sets a preset distribution of blocks from blockfile'
      print '(a)', '--blockfile [FILENAME]  sets the blockfile in FILENAME'
      print '(a)', ''
      print '(a)', ''
    end subroutine print_help

end program blockmcmc

