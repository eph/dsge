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

    double precision :: alpha0_cycle(self%nvar, self%nvar), alpha1_cycle(self%nvar, self%nvar), beta0_cycle(self%nvar, self%nshock)
    double precision :: alphaC_cycle(self%nvar, self%nvar), alphaF_cycle(self%nvar, self%nvar), alphaB_cycle(self%nvar, self%nvar), betaS_cycle(self%nvar, self%nshock)
    double precision :: alpha0_trend(self%nvar, self%nvar), alpha1_trend(self%nvar, self%nvar), betaV_trend(self%nvar, self%nval)
    double precision :: alphaC_trend(self%nvar, self%nvar), alphaF_trend(self%nvar, self%nvar), alphaB_trend(self%nvar, self%nvar)
    double precision :: value_gammaC(self%nval, self%nval), value_gamma(self%nval, self%nval),value_Cx(self%nval, self%nvar), value_Cs(self%nval, self%nshock)
    double precision :: P(self%nshock, self%nshock), R(self%nshock, self%neps)
    double precision :: DD2(self%nobs,1)
    integer :: info

    integer :: k, nvar, nval, nshock
    double precision :: A_cycle(self%nvar, self%nvar), B_cycle(self%nvar, self%nshock)
    double precision :: A_cycle_new(self%nvar, self%nvar), B_cycle_new(self%nvar, self%nshock)
    double precision :: A_trend(self%nvar, self%nvar), B_trend(self%nvar, self%nval)
    double precision :: A_trend_new(self%nvar, self%nvar), B_trend_new(self%nvar, self%nval)
    double precision :: temp1(self%nvar, self%nvar)
    integer, dimension(self%nvar) :: ipiv
    real(8), allocatable :: work(:)
    integer :: lwork
    lwork = self%nvar * self%nvar
    allocate(work(lwork))

    error = 0

    DD2 = 0.0d0

    self%QQ = 0.0d0
    self%ZZ = 0.0d0
    self%HH = 0.0d0

    {system}

    ! Initial calculations for A_cycle, B_cycle, A_trend, B_trend using LAPACK
    call dgetrf(self%nvar, self%nvar, alpha0_cycle, self%nvar, ipiv, info)
    call dgetri(self%nvar, alpha0_cycle, self%nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, alpha0_cycle, self%nvar, alpha1_cycle, self%nvar, 0.0d0, A_cycle, self%nvar)
    call dgemm('N', 'N', self%nvar, self%nshock, self%nvar, 1.0d0, alpha0_cycle, self%nvar, beta0_cycle, self%nvar, 0.0d0, B_cycle, self%nvar)

    call dgetrf(self%nvar, self%nvar, alpha0_trend, self%nvar, ipiv, info)
    call dgetri(self%nvar, alpha0_trend, self%nvar, ipiv, work, lwork, info)
    call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, alpha0_trend, self%nvar, alpha1_trend, self%nvar, 0.0d0, A_trend, self%nvar)
    call dgemm('N', 'N', self%nvar, self%nval, self%nvar, 1.0d0, alpha0_trend, self%nvar, betaV_trend, self%nvar, 0.0d0, B_trend, self%nvar)

    ! ! Main loop for k
    do k = 1, {k}
    !     ! Calculations for A_cycle_new
         temp1 = 0.0d0
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, -1.0d0, alphaF_cycle, self%nvar, A_cycle, self%nvar, 1.0d0, temp1, self%nvar)
         temp1 = temp1 + alphaC_cycle
         call dgetrf(self%nvar, self%nvar, temp1, self%nvar, ipiv, info)
         call dgetri(self%nvar, temp1, self%nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, temp1, self%nvar, alphaB_cycle, self%nvar, 0.0d0, A_cycle_new, self%nvar)
    !
    !     ! ... (continue with all other calculations)
         ! Calculations for B_cycle_new

         B_cycle_new = matmul(temp1, matmul(alphaF_cycle, matmul(B_cycle, P)) + betaS_cycle)

         ! Calculations for A_trend_new
         temp1 = 0.0d0
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, -1.0d0, alphaF_trend, self%nvar, A_trend, self%nvar, 1.0d0, temp1, self%nvar)
         temp1 = temp1 + alphaC_trend
         call dgetrf(self%nvar, self%nvar, temp1, self%nvar, ipiv, info)
         call dgetri(self%nvar, temp1, self%nvar, ipiv, work, lwork, info)
         call dgemm('N', 'N', self%nvar, self%nvar, self%nvar, 1.0d0, temp1, self%nvar, alphaB_trend, self%nvar, 0.0d0, A_trend_new, self%nvar)
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
self%RR(1:nvar, 1:self%neps) = matmul(B_cycle, R)

! Second block
self%RR(nvar+1:2*nvar, 1:self%neps) = matmul(B_cycle, R)

! Third block
! Already initialized to zero, so no operation needed for zeroS

! Fourth block
! zeroV.T @ zeroS will be zero, so no operation needed here either

! Fifth block
self%RR(3*nvar+nval+1:3*nvar+nval+nshock, 1:self%neps) = R

!call write_array_to_file('TT.txt',self%TT)
!call write_array_to_file('RR.txt',self%RR)
error=0

    self%DD = DD2(:,1)




    if (info==1) error = 0

  end subroutine system_matrices


end module model_t
