
module model_t
  use, intrinsic :: iso_fortran_env, only: wp => real64

  use gensys, only: do_gensys
  use fortress, only : fortress_lgss_model
  use fortress_prior_t, only: model_prior => prior
  use fortress_linalg, only: Kronecker, inverse, cholesky

  use mg, only: find_max_it_fortran, solve_ma_alt_fortran

  implicit none

  type, public, extends(fortress_lgss_model) :: model
     integer :: neq = {nendo_vars}

     real(wp), allocatable :: THETA(:,:), ALPHA(:,:)
     integer, allocatable :: obs_ind(:)


   contains
     procedure :: max_it
     procedure :: system_matrices

     procedure :: lik => lik_direct

  end type model


  interface model
     module procedure new_model
  end interface model


contains

  type(model) function new_model() result(self)

    character(len=144) :: name, datafile, priorfile
    integer :: nobs, T, ns, npara, neps

    name = '{name}'
    datafile = 'data.txt'
    priorfile = 'prior.txt'

    nobs = {yy.shape[1]}
    T = {yy.shape[0]}

    ns = 10 !
    npara = {npara}
    neps = {neps}

    call self%construct_model(name, datafile, priorfile, npara, nobs, T, ns, neps)

    allocate(self%obs_ind(self%nobs))

    self%p0 = [{pmsv}]
    !self%p0 = {pmsv}
    ! self%neta = 4
    self%t0 = 0
    self%HH = 0.0_wp
  end function new_model

  integer function max_it(self, para)

    class(model), intent(inout) :: self

    real(wp), intent(in) :: para(self%npara)

    call find_max_it_fortran(j_matrices, para, self%npara, self%neq, self%neps, max_it)

  contains

    subroutine j_matrices(para, j, Aj, Bj, Cj, Fj, Gj)

      real(wp), intent(in) :: para(self%npara)
      integer, intent(in) :: j

      real(wp), intent(out) :: Aj(self%neq, self%neq), Bj(self%neq, self%neq)
      real(wp), intent(out) :: Cj(self%neq, self%neq), Fj(self%neq, self%neps)
      real(wp), intent(out) :: Gj(self%neq, self%neps)


      {j_matrices}


    end subroutine j_matrices


  end function max_it



  subroutine system_matrices(self, para, error)

    class(model), intent(inout) :: self
    real(wp), intent(in) :: para(self%npara)

    integer, intent(out) :: error 

    real(wp) :: A(self%neq, self%neq), B(self%neq, self%neq)
    real(wp) :: C(self%neq, self%neq), F(self%neq, self%neps)
    real(wp) :: G(self%neq, self%neps), N(self%neps, self%neps)

    real(wp) :: Ainf(self%neq, self%neq), Binf(self%neq, self%neq)
    real(wp) :: Cinf(self%neq, self%neq), Finf(self%neq, self%neps)
    real(wp) :: Ginf(self%neq, self%neps), DD2(self%nobs,1)
    
    integer :: max_it
    real(wp) :: BETA(self%neq, self%neps)
    real(wp), allocatable :: MA_VECTOR(:,:)

    error = 1

    {zero_matrices}

    {inf_matrices}


    max_it = self%max_it(para)

    !self%QQ = 1.0d0
    N = 0.0d0

    if (allocated(self%THETA)) deallocate(self%THETA)
    if (allocated(self%ALPHA)) deallocate(self%ALPHA)
    allocate(self%THETA((max_it+1)*self%neq, self%neps))
    allocate(self%ALPHA(self%neq, self%neq))
    call solve_ma_alt_fortran(A,B,C,F,G,N,j_matrices, &
         Ainf,Binf,Cinf,Finf,Ginf, para, &
         max_it, self%npara, self%neq, self%neps, &
         self%THETA, self%ALPHA, BETA, error)
    !print*,'error = ', error
    !deallocate(MA_VECTOR)

    self%DD = DD2(:,1)
    
  contains

    subroutine j_matrices(para, j, Aj, Bj, Cj, Fj, Gj)

      real(wp), intent(in) :: para(self%npara)
      integer, intent(in) :: j

      real(wp), intent(out) :: Aj(self%neq, self%neq), Bj(self%neq, self%neq)
      real(wp), intent(out) :: Cj(self%neq, self%neq), Fj(self%neq, self%neps)
      real(wp), intent(out) :: Gj(self%neq, self%neps)

      {j_matrices}



    end subroutine j_matrices



  end subroutine system_matrices

  real(wp) function lik_direct(self, para, T) result(loglik)

  class(model), intent(inout) :: self
  real(wp), intent(in) :: para(self%npara)
  integer, intent(in), optional :: T


  real(wp) ::  ALPHA(self%neq,self%neq), BETA(self%neq,self%neps), obs_sel(self%nobs), OMEGA(self%neps,self%neps)
  real(wp), allocatable :: PSI(:,:), THETAcOMEGA(:,:)
  real(wp), allocatable :: PSI_SHIFT(:,:), GAMMAx(:,:), this_GAMMA(:,:), COV(:,:), effCOV(:,:), effyvec(:)
  real(wp) :: THET1(self%neq,self%neps), ALPHA_THET1(self%neq,self%neps)
  integer :: max_it, RC,i, info

  real(wp) :: yvec(self%T*self%nobs,1), det

  real(wp) :: ddot
  

  integer :: max_emp_it, neffobs, j,error !, ipiv(self%nobs*T)

  integer, allocatable :: effsel(:), ipiv(:)

  integer :: use_T
            
  use_T = self%T
  if (present(T)) use_T = T

  ! if (self%inbounds(para) .neqv. .true.) then
  !    loglik = -1000000000000.0_wp
  !    return
  ! end if
  ! 
  max_it = self%max_it(para)
  max_emp_it = max_it + 300

  !obs_ind = self%get_obs()

  allocate(THETAcOMEGA((max_it+1)*self%neq,self%neps))
  allocate(PSI(self%nobs,self%neps*max_emp_it))


  call self%system_matrices(para, error)

  ! safety value
  if (error > 0) then
     loglik = -1000000000.0_wp
     deallocate(THETAcOMEGA,PSI)
     return
  end if

  !------------------------------------------------------------
  ! Set THETA -> THETA*chol(OMEGA)
  !------------------------------------------------------------
  OMEGA = self%QQ
  call dpotrf('L', self%neps, OMEGA, self%neps, info) ! OMEGA -> chol(OMEGA)
  call dgemm('n','n',(max_it+1)*self%neq,self%neps,self%neps,1.0_wp, &
       self%THETA,(max_it+1)*self%neq, &
       OMEGA,self%neps,0.0_wp, &
       THETAcOMEGA,(max_it+1)*self%neq)


  THET1 = THETAcOMEGA(max_it*self%neq+1:(max_it+1)*self%neq,:);

  self%obs_ind = {obs_ind}
  !self%obs_ind = [1,2,3,4]
  !------------------------------------------------------------
  ! PSI is set up as the MA representation of the 
  !   observables, stacked horizontally.
  !
  ! PSI = [ THET_[1,y] ... THET[n,y] ] 
  !------------------------------------------------------------
  PSI = 0.0_wp
  do i = 1, (max_it+1)
     PSI(:,(i-1)*self%neps+1:i*self%neps) = THETAcOMEGA(self%obs_ind+(i-1)*self%neq,:)
  end do


  do i = max_it+2,max_emp_it 
     call dgemm('n','n',self%neq,self%neps,self%neq,1.0_wp,self%ALPHA,self%neq,THET1,self%neq,0.0_wp,ALPHA_THET1,self%neq)
     PSI(:,(i-1)*self%neps+1:i*self%neps) = ALPHA_THET1(self%obs_ind,:)
     THET1 = ALPHA_THET1
  end do


  !------------------------------------------------------------
  ! Compute the autocovariances...
  !------------------------------------------------------------
 

  allocate(GAMMAx(self%nobs*use_T,self%nobs),  &
           PSI_SHIFT(self%nobs,self%neps*max_emp_it), &
           this_GAMMA(self%nobs*use_T,self%nobs), &
           COV(self%nobs*use_T,self%nobs*use_T))

  GAMMAx = 0.0_wp

  !print*,'psi[1,1]=',GAMMA(1,1)

  do i = 1,use_T
     PSI_SHIFT = 0.0_wp
     PSI_SHIFT(:,(i-1)*self%neps+1:self%neps*max_emp_it) = PSI(:,1:(max_emp_it-i+1)*self%neps)
     !     print*,(max_emp_it-i+1)*neps
     call dgemm('n','t',self%nobs,self%nobs,self%neps*max_emp_it,1.0_wp,PSI,self%nobs,PSI_SHIFT,self%nobs,0.0_wp,GAMMAx((i-1)*self%nobs+1:i*self%nobs,:),self%nobs)
  end do

  !print*,'gamma[6,3]=',GAMMA(1,1)
  !do i=1,ny
  !   write(*,'(7f9.4)'),GAMMA((nobs-1)*ny+i,:)
  !end do
  this_GAMMA = GAMMAx
  
  ! ! measurement errors
  this_GAMMA(1:self%nobs,1:self%nobs) = this_GAMMA(1:self%nobs,1:self%nobs) + self%HH

  COV = 0.0_wp
  COV(1:use_T*self%nobs,1:self%nobs) = this_GAMMA

  do i = 2, use_T
     this_GAMMA(self%nobs+1:self%nobs*use_T,:) = this_GAMMA(1:(use_T-1)*self%nobs,:)
     this_GAMMA(1:self%nobs,:) = transpose(GAMMAx((i-1)*self%nobs+1:i*self%nobs,:))

     COV(:,(i-1)*self%nobs+1:i*self%nobs) = this_GAMMA
  end do

    ! !------------------------------------------------------------
    ! ! Adjust for missing data
    ! !------------------------------------------------------------
    ! yvec = get_data(para)
    ! !  print*,'yvec[1,1] = ',yvec(1,1)
    
    yvec=0.0_wp
    do i = 1, use_T
       yvec((i-1)*self%nobs+1:i*self%nobs,1) = self%yy(:,i) - self%DD
    end do

    neffobs = self%nobs*use_T - count(isnan(yvec(1:use_T*self%nobs,1)))

    allocate(effsel(neffobs),effCOV(neffobs,neffobs),effyvec(neffobs),ipiv(neffobs))

    j = 1

    do i = 1,use_T*self%nobs
       if (isnan(yvec(i,1)) .eqv. .false.) then 
          effsel(j) = i
          effyvec(j) = yvec(i,1)
          j = j + 1
       end if
    end do

    effCOV = COV(effsel,effsel)
    call cholesky(effCOV, info)
    !call dpotrf('L', neffobs, effCOV, neffobs, info) ! OMEGA -> chol(OMEGA)

    if (info .ne. 0) then
       print*,'Cannot invert covariance matrix'

       loglik = -100000000000.0_wp

       deallocate(effsel,effCOV,effyvec)
       deallocate(THETAcOMEGA)
       deallocate(PSI)
       deallocate(GAMMAx,PSI_SHIFT,this_GAMMA,COV)
       return


    end if
    ! compute determinate
    det = 0.0_wp
    do i = 1,neffobs
       det = det + log(effCOV(i,i))
    end do


    ! ! eliminate the upper trianngle
    ! do i = 2,neffobs
    !    effCOV(1:i-1,i) = 0.0_wp    
    ! end do

    !call inverse(effCOV, info)
    !effyvec = matmul(effCOV,effyvec)

    ! !print*,COV(1,1)
    ! ! now solve chol(Omega)*x = yvec
    call dgetrf(neffobs,neffobs,effCOV,neffobs,ipiv,info)
    call dgetrs('n',neffobs,1,effCOV,neffobs,ipiv,effyvec,neffobs,info)


    if (info .ne. 0) then
       loglik = -100000000000.0_wp

       deallocate(effsel,effCOV,effyvec,ipiv)
       deallocate(THETAcOMEGA)
       deallocate(PSI)
       deallocate(GAMMAx,PSI_SHIFT,this_GAMMA,COV)

       return
    end if


    loglik = ddot(neffobs,effyvec,1,effyvec,1)
    loglik = -0.5_wp*neffobs*log(2.0_wp*3.141592653589793_wp) - det - 0.5_wp*loglik


    deallocate(effsel,effCOV,effyvec,ipiv)
    deallocate(THETAcOMEGA)
    deallocate(PSI)

    deallocate(GAMMAx,PSI_SHIFT,this_GAMMA,COV)
           


end function lik_direct

end module model_t



