module model_t
  use, intrinsic :: iso_fortran_env, only: wp => real64

  use gensys, only: do_gensys
  use fortress, only : fortress_lgss_model
  use fortress_prior_t, only: model_prior => prior

  implicit none

  type, public, extends(fortress_lgss_model) :: model
     integer :: nvar, nshock, nval

   contains
     procedure :: system_matrices
  end type model


  interface model
     module procedure new_model
  end interface model


contains

  type(model) function new_model() result(self)

    character(len=144) :: name, datafile, priorfile
    integer :: nobs, T, ns, npara, neps

    name = 'fhp'
    datafile = 'data.txt'
    priorfile = 'prior.txt'

    nobs = {cmodel.yy.shape[1]}
    T = {cmodel.yy.shape[0]}

    self%nvar = {len(model['variables'])}
    self%nval = {len(model['values'])}
    self%nshock = {len(model['shocks'])}
    ns = 3*self%nvar + self%nval + self%nshock
    npara = {len(model['parameters'])}
    neps = {len(model['innovations'])}

    call self%construct_model(name, datafile, priorfile, npara, nobs, T, ns, neps)

!    self%p0 = {p0}

    self%t0 = {t0}
  end function new_model

  subroutine system_matrices(self, para, error)

    class(model), intent(inout) :: self
    real(wp), intent(in) :: para(self%npara)

    integer, intent(out) :: error

    ! Declare dimensions first so they can be used in array bounds
    integer :: k, nvar, nval, nshock, nobs, neps
    integer :: info
    integer :: lwork
    integer, dimension(:), allocatable :: ipiv
    real(8), allocatable :: work(:)

    ! Now use the dimension variables in array declarations
    double precision, allocatable :: alpha0_cycle(:,:), alpha1_cycle(:,:), beta0_cycle(:,:)
    double precision, allocatable :: alphaC_cycle(:,:), alphaF_cycle(:,:), alphaB_cycle(:,:), betaS_cycle(:,:)
    double precision, allocatable :: alpha0_trend(:,:), alpha1_trend(:,:), betaV_trend(:,:)
    double precision, allocatable :: alphaC_trend(:,:), alphaF_trend(:,:), alphaB_trend(:,:)
    double precision, allocatable :: value_gammaC(:,:), value_gamma(:,:), value_Cx(:,:), value_Cs(:,:)
    double precision, allocatable :: P(:,:), R(:,:)
    double precision, allocatable :: DD2(:,:)
    double precision, allocatable :: A_cycle(:,:), B_cycle(:,:)
    double precision, allocatable :: A_cycle_new(:,:), B_cycle_new(:,:)
    double precision, allocatable :: A_trend(:,:), B_trend(:,:)
    double precision, allocatable :: A_trend_new(:,:), B_trend_new(:,:)
    double precision, allocatable :: temp1(:,:)

    ! Set dimensions from self
    nvar = self%nvar
    nval = self%nval
    nshock = self%nshock
    nobs = self%nobs
    neps = self%neps
    lwork = nvar * nvar

    ! Allocate all arrays
    allocate(alpha0_cycle(nvar, nvar), alpha1_cycle(nvar, nvar), beta0_cycle(nvar, nshock))
    allocate(alphaC_cycle(nvar, nvar), alphaF_cycle(nvar, nvar), alphaB_cycle(nvar, nvar), betaS_cycle(nvar, nshock))
    allocate(alpha0_trend(nvar, nvar), alpha1_trend(nvar, nvar), betaV_trend(nvar, nval))
    allocate(alphaC_trend(nvar, nvar), alphaF_trend(nvar, nvar), alphaB_trend(nvar, nvar))
    allocate(value_gammaC(nval, nval), value_gamma(nval, nval), value_Cx(nval, nvar), value_Cs(nval, nshock))
    allocate(P(nshock, nshock), R(nshock, neps))
    allocate(DD2(nobs, 1))
    allocate(A_cycle(nvar, nvar), B_cycle(nvar, nshock))
    allocate(A_cycle_new(nvar, nvar), B_cycle_new(nvar, nshock))
    allocate(A_trend(nvar, nvar), B_trend(nvar, nval))
    allocate(A_trend_new(nvar, nvar), B_trend_new(nvar, nval))
    allocate(temp1(nvar, nvar))
    allocate(ipiv(nvar))
    allocate(work(lwork))

    error = 0

    DD2 = 0.0d0

    self%QQ = 0.0d0
    self%ZZ = 0.0d0
    self%HH = 0.0d0

    {system}

    ! Initial calculations for A_cycle, B_cycle, A_trend, B_trend using LAPACK
    call dgetrf(nvar, nvar, alpha0_cycle, nvar, ipiv, info)
    call dgetri(nvar, alpha0_cycle, nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', nvar, nvar, nvar, 1.0d0, alpha0_cycle, nvar, alpha1_cycle, nvar, 0.0d0, A_cycle, nvar)
    call dgemm('N', 'N', nvar, nshock, nvar, 1.0d0, alpha0_cycle, nvar, beta0_cycle, nvar, 0.0d0, B_cycle, nvar)

    call dgetrf(nvar, nvar, alpha0_trend, nvar, ipiv, info)
    call dgetri(nvar, alpha0_trend, nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', nvar, nvar, nvar, 1.0d0, alpha0_trend, nvar, alpha1_trend, nvar, 0.0d0, A_trend, nvar)
    call dgemm('N', 'N', nvar, nval, nvar, 1.0d0, alpha0_trend, nvar, betaV_trend, nvar, 0.0d0, B_trend, nvar)

    ! ! Main loop for k
    do k = 1, {k}
    !     ! Calculations for A_cycle_new
         temp1 = 0.0d0
         call dgemm('N', 'N', nvar, nvar, nvar, -1.0d0, alphaF_cycle, nvar, A_cycle, nvar, 1.0d0, temp1, nvar)
         temp1 = temp1 + alphaC_cycle
         call dgetrf(nvar, nvar, temp1, nvar, ipiv, info)
         call dgetri(nvar, temp1, nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', nvar, nvar, nvar, 1.0d0, temp1, nvar, alphaB_cycle, nvar, 0.0d0, A_cycle_new, nvar)
    !
    !     ! ... (continue with all other calculations)
         ! Calculations for B_cycle_new

         B_cycle_new = matmul(temp1, matmul(alphaF_cycle, matmul(B_cycle, P)) + betaS_cycle)

         ! Calculations for A_trend_new
         temp1 = 0.0d0
         call dgemm('N', 'N', nvar, nvar, nvar, -1.0d0, alphaF_trend, nvar, A_trend, nvar, 1.0d0, temp1, nvar)
         temp1 = temp1 + alphaC_trend
         call dgetrf(nvar, nvar, temp1, nvar, ipiv, info)
         call dgetri(nvar, temp1, nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', nvar, nvar, nvar, 1.0d0, temp1, nvar, alphaB_trend, nvar, 0.0d0, A_trend_new, nvar)
         B_trend_new = matmul(temp1, matmul(alphaF_trend, B_trend))

         ! Updating variables
         A_cycle = A_cycle_new
         B_cycle = B_cycle_new
         A_trend = A_trend_new
         B_trend = B_trend_new

    !
    end do
 !   call write_array_to_file('A_cycle.txt', A_cycle)
 !   call write_array_to_file('B_cycle.txt', B_cycle)
 !   call write_array_to_file('A_trend.txt', A_trend)
 !   call write_array_to_file('B_trend.txt', B_trend)

    self%DD = DD2(:,1)

    self%TT = 0.0d0
    self%RR = 0.0d0

    ! First block row
    nvar = self%nvar
    nshock = self%nshock
    nval = self%nval

self%TT(1:nvar, 1:nvar) = matmul(matmul(B_trend, value_gamma), value_Cx)
self%TT(1:nvar, (nvar+1):2*nvar) = A_cycle
self%TT(1:nvar, (2*nvar+1):(3*nvar)) = A_trend
self%TT(1:nvar, (3*nvar+1):(3*nvar+nval)) = matmul(B_trend, value_gammaC)
self%TT(1:nvar, (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(B_cycle, P) + matmul(matmul(B_trend, value_gamma), value_Cs)

! Second block row
self%TT((nvar+1):(2*nvar), (nvar+1):(2*nvar)) = A_cycle
self%TT((nvar+1):(2*nvar), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(B_cycle, P)

! Third block row
self%TT((2*nvar+1):(3*nvar), 1:nvar) = matmul(matmul(B_trend, value_gamma), value_Cx)
self%TT((2*nvar+1):(3*nvar), (2*nvar+1):(3*nvar)) = A_trend
self%TT((2*nvar+1):(3*nvar), (3*nvar+1):(3*nvar+nval)) = matmul(B_trend, value_gammaC)
self%TT((2*nvar+1):(3*nvar), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(matmul(B_trend, value_gamma), value_Cs)

! Fourth block row
self%TT((3*nvar+1):(3*nvar+nval), 1:nvar) = matmul(value_gamma, value_Cx)
self%TT((3*nvar+1):(3*nvar+nval), (3*nvar+1):(3*nvar+nval)) = value_gammaC
self%TT((3*nvar+1):(3*nvar+nval), (3*nvar+nval+1):(3*nvar+nval+nshock)) = matmul(value_gamma, value_Cs)

! Fifth block row
self%TT((3*nvar+nval+1):(3*nvar+nval+nshock), (3*nvar+nval+1):(3*nvar+nval+nshock)) = P

! Assuming self%RR is already initialized to zero and its dimensions are set correctly
! nvar, nshock, neps are already defined

! First block
self%RR(1:nvar, 1:neps) = matmul(B_cycle, R)

! Second block
self%RR(nvar+1:2*nvar, 1:neps) = matmul(B_cycle, R)

! Third block
! Already initialized to zero, so no operation needed for zeroS

! Fourth block
! zeroV.T @ zeroS will be zero, so no operation needed here either

! Fifth block
self%RR(3*nvar+nval+1:3*nvar+nval+nshock, 1:neps) = R

!call write_array_to_file('TT.txt',self%TT)
!call write_array_to_file('RR.txt',self%RR)
error=0

    self%DD = DD2(:,1)




    if (info==1) error = 0

  end subroutine system_matrices


end module model_t
