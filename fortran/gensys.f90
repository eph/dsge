! gensys.f90 -- A module for implementing Chris Sims' GENSYS
!  This code is based on Chris Sims' MATLAB code &
!  Iskander Karibzhanov's C + Intel MKL implementation
module gensys

  use mkl95_precision, only: wp => dp
  use omp_lib


  implicit none

  real, parameter :: verysmall = 0.000001_wp

  complex(wp), parameter :: CPLX_ZERO = dcmplx(0.0_wp,0.0_wp)
  complex(wp), parameter :: CPLX_ONE = dcmplx(1.0_wp, 0.0_wp)
  complex(wp), parameter :: CPLX_NEGONE = dcmplx(-1.0_wp, 0.0_wp)

  integer :: nunstab = 0
  integer :: zxz = 0 
  integer :: fixdiv = 1
  real(wp) :: stake = 1.01_wp

  !$OMP THREADPRIVATE(nunstab,stake,zxz,fixdiv)

contains

  subroutine do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, &
       G0, G1, C0, PSI, PI, DIV)
    
    implicit none 

    real(wp), intent(inout) :: G0(:,:), G1(:,:), C0(:), PSI(:,:), PI(:,:), div, loose
    real(wp), intent(out) :: TT(size(G0,1),size(G0,1)), CC(size(G0,1)), RR(size(G0,1),size(PSI,2)), fmat, fwt, ywt, gev
    !f2py depend(size(G0,1)) TT, C0, RR 
    complex(wp), dimension(size(G0,1), size(G0, 1)) :: Q, Z, AA, BB, cG0,cG1
    complex(wp), dimension(size(G0,1)) :: alpha, beta
    complex(wp) :: cRR(size(G0,1), size(PSI,2))
    integer, intent(out) :: eu(2)
    complex(wp), allocatable :: etawt(:,:), zwt(:,:)
    integer :: info, pin, n, i, ipiv(size(G0,1)), ldzt, nstab

    complex(wp), allocatable :: Qstab(:,:), Qunstab(:,:)
    ! svd  stuff
    complex(wp), allocatable :: eta_u(:,:), eta_v(:,:), zwt_u(:,:), zwt_v(:,:)
    real(wp), allocatable :: eta_s(:), zwt_s(:)
    complex(wp), allocatable :: zwt_u_tran(:,:), int_mat(:,:), int_mat2(:,:), tmat(:,:), vv2(:,:), cPI(:,:)

    integer :: ldvt, lmin, nbigev
    logical :: unique
    real(wp) :: norm

    !test stuff
    real(wp), dimension(50,50) :: sreal, simag, treal, timag, qreal, qimag, zreal, zimag


    eu      = (/0, 0/)
    nbigev  = 0
    n       = size(G0,1)
    pin     = size(PI,2)
    zxz     = 0
    nunstab = 0
    stake   = 1.01_wp
    fixdiv  = 1

    fmat = 0 
    ywt = 0
    gev = 0
    fwt = 0

    TT = 0.0_wp
    CC = 0.0_wp
    RR = 0.0_wp

    allocate(cPI(n, pin))

    cPI = dcmplx(PI)

    call qz(G0, G1, AA, BB, Q, Z, alpha, beta, n, info)
    Q = transpose(conjg(Q)) 


    if (zxz == 1) then
       print *, "Coincident zeros. Indeterminacy and/or nonexistance."
       eu = -2
       deallocate(cPI)
       return
    end if


    nstab = n - nunstab

    if (nstab == 0) then
       eu = -2
       deallocate(cPI)
       return
    end if

    allocate(Qstab(nstab, n), Qunstab(nunstab, n))

    Qstab = Q(1:nstab, :)
    Qunstab = Q(nstab+1:n,:)



    ! etawt = Q2*PI
    allocate(etawt(nunstab, pin)) 

    call zgemm('n','n', nunstab, pin, n, CPLX_ONE, Qunstab, nunstab, & 
         cPI, n, CPLX_ZERO, etawt, nunstab)
    lmin = min(nunstab, pin)

    allocate(eta_u(nunstab, lmin), eta_s(lmin), eta_v(lmin, pin))
    call zsvd(etawt, eta_u, eta_s, eta_v, nunstab, pin, lmin)

    do i = 1,size(eta_s,1)
       if (eta_s(i) > verysmall) nbigev = nbigev + 1
    end do

    if (nbigev >= nunstab) eu(1) = 1

   ! zwt
    allocate(zwt(nstab, pin))
    call zgemm('n','n', nstab, pin, n, CPLX_ONE, Qstab, nstab, & 
         cPI, n, CPLX_ZERO, zwt, nstab)

    ldzt = min(nstab, pin)
    allocate(zwt_u(nstab, ldzt), zwt_s(ldzt), zwt_v(ldzt, pin))
    call zsvd(zwt, zwt_u, zwt_s, zwt_v, nstab, pin, ldzt)

    ! Check for uniques
    if (size(zwt_v)==0) then
       unique = .true.
    else
       allocate(vv2(ldzt, pin))!,eta_v_squared(lmin,lmin))
!!$       vv2 = zwt_v
!!$       call zgemm('n','c',lmin,lmin,pin,CPLX_ONE,eta_v,lmin,eta_v,lmin,CPLX_ZERO,eta_v_squared,lmin)
!!$       call zgemm('n','n',ldzt,pin,lmin,-CPLX_NEGONE,eta_v_squared,lmin,zwt_v,lmin,CPLX_ONE,vv2,lmin)
       ! this needs to be put into LAPACK
       vv2 = zwt_v - matmul(matmul(eta_v,transpose(conjg(eta_v))),zwt_v);
       call compute_norm(matmul(transpose(vv2), vv2), norm, size(vv2, 2), size(vv2, 2))
       !print*,norm,'fdsafa'
       unique = norm < n*verysmall;
    endif
    
!    TT(1,5)= zwt(2,1)
!    deallocate(Qstab, Qunstab, cPI, etawt, eta_u, eta_s, eta_v, zwt, zwt_u, zwt_s, zwt_v, vv2)
!    return
    if (unique) then
       eu(2) = 1
    else
       !print*,'Indeterminancy'
       eu(2) = 0
    endif

    ! eta_v => deta/veta' (recall zsvd returns v', not v)
    do i = 1, lmin
       call zdscal(lmin, 1.0_wp/eta_s(i), eta_v(i,:), 1)
    end do

    ! zwt_u_tran => deta1*uu'
!!$    allocate(zwt_u_tran(ldzt, nstab))
!!$
!!$    zwt_u_tran = transpose(conjg(eta_u))
!!$
!!$    do i = 1, ldzt
!!$       call zdscal(nstab, zwt_s(i), zwt_u_tran(i,:), 1)
!!$    end do
 
 
    allocate(tmat(nstab, n), int_mat(lmin, nstab))
    tmat = 0.0_wp
    do i  = 1, nstab
       tmat(i,i) = 1.0_wp
    end do
 
    ! int mat = deta\veta'*veta1*deta1*ueta1'
    call zgemm('n','c', lmin, nstab, pin, CPLX_ONE, eta_v, lmin, zwt, nstab, &
         CPLX_ZERO, int_mat, lmin)
    call zgemm('c','c', nstab, nunstab, lmin, CPLX_NEGONE, int_mat, lmin, &
         eta_u, nunstab, CPLX_ZERO, tmat(:, nstab+1:n), nstab)
 
 
    cG0 = cmplx(0.0_wp,0.0_wp)
    cG0(1:(n-nunstab),:) = matmul(tmat,AA)
 
    do i = n-nunstab+1,n
       cG0(i,i) = dcmplx(1.0_wp, 0.0_wp)
    end do
 
    cG1 = dcmplx(0.0_wp,0.0_wp)
    cG1(1:(n-nunstab),:) = matmul(tmat,BB) 
 
    call zgesv(n, n, cG0, n, ipiv, cG1, n, info)
 
    TT = real(matmul(matmul(Z, cG1), transpose(conjg(Z))));
    cRR = dcmplx(0.0_wp)
    cRR(1:n-nunstab,:) = matmul(matmul(tmat,Q),PSI)
 
    cG0 = cmplx(0.0_wp,0.0_wp)
    cG0(1:(n-nunstab),:) = matmul(tmat,AA)
 
    do i = n-nunstab+1,n
       cG0(i,i) = cmplx(1.0_wp, 0.0_wp)
    end do
    
    call zgesv(n, size(RR, 2), cG0, n, ipiv, cRR, n, info)
    RR = real(matmul(Z,cRR))
 
    deallocate(etawt, eta_u, eta_s, eta_v)
    deallocate(zwt, zwt_u, zwt_s, zwt_v)!, zwt_u_tran)
    deallocate(int_mat, tmat, vv2, cPI, Qstab, Qunstab)

    call mkl_free_buffers()
  end subroutine do_gensys

  subroutine compute_norm(d, norm, m, n)
    ! computes 2-norm of matrix d [m x n]
    complex(wp), intent(in) :: d(m, n)
    real(wp), intent(out) :: norm
    integer, intent(in) :: m, n

    integer :: md, lwork, info
    real(wp), allocatable :: rwork(:), norm_m(:)
    complex(wp), allocatable :: work(:)

    md = minval((/ m, n /),1)
    lwork = -1
    norm = 10.0_wp
    allocate(rwork(5*md), norm_m(md), work(100))
    call zgesvd('N','N', m, n, d, m, norm_m, 0, m, 0, md, work, lwork, rwork, info)
    lwork = work(1)
    deallocate(work)

    allocate(work(lwork))
    call zgesvd('N','N', m, n, d, m, norm_m, 0, m, 0, md, work, lwork, rwork, info)
    if (info < 0) then
       print*,'bad value'
       deallocate(work, rwork, norm_m)
       return
    end if 

    norm = sqrt(maxval(norm_m))

    deallocate(work, rwork, norm_m)
  end subroutine compute_norm
    
  logical function delctg(alpha, beta)

    complex(wp), intent(in) :: alpha, beta
    real(wp) :: A, B, divhat

    A = sqrt(real(alpha)**2 + aimag(alpha)**2)
    B = abs(real(beta))!sqrt(real(beta)**2.0 + aimag(beta)**2.0)

    if (A > 0.0_wp) then
       divhat = B/A
       if (((fixdiv == 1) .and. (1.0_wp + verysmall < divhat)) .and. divhat < stake) then
          stake = (1.0_wp + divhat) / 2.0_wp
       end if

    end if

    if (A < verysmall .and. B < verysmall) then
       zxz = 1
    end if

    if (B > (stake*A)) then
       nunstab = nunstab + 1
       delctg = .false.
    else
       delctg = .true.
    end if

  end function delctg



  subroutine qz(a, b, aa, bb, q, z, alpha, beta, n, info)
    
    double precision, intent(in) :: a(n,n), b(n,n)
    complex(wp), intent(out), dimension(n,n) :: q, z, aa, bb
    complex(wp), intent(out) :: alpha(n), beta(n)
    integer, intent(out) :: info

    integer :: n, sdim, lwork, i
    complex(wp), dimension(n, n) :: cplxa, cplxb
    complex(wp), allocatable :: work(:)
    double precision, dimension(8*n) :: rwork
    logical, dimension(4*n) :: bwork

    cplxa = dcmplx(a)
    cplxb = dcmplx(b)
    
    allocate(work(33))
    lwork = -1
    call zgges('V','V','S', delctg, n, cplxa, n, cplxb, n, sdim, alpha, beta, q, &
         n, z, n, work, lwork, rwork, bwork, info)
    if (info < 0) then
       print*,'zgges: input ', -info, 'had an illegal value.'
       deallocate(work)
       return 
    endif
    lwork =  int(work(1))
    deallocate(work)

    allocate(work(lwork))
    call zgges('V','V','S', delctg, n, cplxa, n, cplxb, n, sdim, alpha, beta, q, &
         n, z, n, work, lwork, rwork, bwork, info)
    deallocate(work)

    nunstab = n - sdim
    
    aa = cplxa
    bb = cplxb


  end subroutine qz

  ! wrapper for zgesvd with options 'S' 'S'

  ! Performs decomposition 
  ! A = U*S*V
  subroutine zsvd(A, U, S, V, nrow, ncolumn, nmin)

    integer, intent(in) :: nrow, ncolumn, nmin
    complex(wp), intent(in) :: A(nrow, ncolumn)

    complex(wp), intent(out) :: U(nrow, nmin), V(nmin, ncolumn)
    real(wp), intent(out) :: S(nmin)

    complex(wp) :: AA(nrow, ncolumn)
    integer :: info, lwork
    complex(wp), allocatable :: work(:)
    real(wp) :: rwork(5*nmin)

    AA = A
    ! query workspace
    allocate(work(1))
    lwork = -1
    call zgesvd('S','S', nrow, ncolumn, AA, nrow, S, U, nrow, V, nmin, &
         work, lwork, rwork, info)
    lwork = int(work(1))
    deallocate(work)

    ! compute svd
    allocate(work(lwork))
    call zgesvd('S','S', nrow, ncolumn, AA, nrow, S, U, nrow, V, nmin, &
         work, lwork, rwork, info)
    deallocate(work)

  end subroutine zsvd
end module gensys
