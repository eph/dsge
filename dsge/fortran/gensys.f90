! gensys.f90 -- A module for implementing Chris Sims' GENSYS
!  This code is based on Chris Sims' MATLAB code &
!  Iskander Karibzhanov's C + Intel MKL implementation
module gensys

  !use mkl95_precision, only: wp => dp
  !use omp_lib
  implicit none
  integer, parameter :: wp = selected_real_kind(15)
  !




  double precision, parameter :: verysmall = 0.000001d0

  double complex, parameter :: CPLX_ZERO = dcmplx(0.0d0,0.0d0)
  double complex, parameter :: CPLX_ONE = dcmplx(1.0d0, 0.0d0)
  double complex, parameter :: CPLX_NEGONE = dcmplx(-1.0d0, 0.0d0)

  integer :: nunstab = 0
  integer :: zxz = 0
  integer :: fixdiv = 1
  double precision :: stake = 1.01d0

  !$OMP THREADPRIVATE(nunstab,stake,zxz,fixdiv)

contains

   subroutine do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, &
        ns, neps, neta, &
        G0, G1, C0, PSI, PI, DIV)

     implicit none
     integer, intent(in) :: ns, neps, neta

     double precision, intent(inout) :: G0(ns,ns), G1(ns,ns), C0(ns), &
          PSI(ns,neps), PI(ns,neta), DIV
     double precision, intent(out) :: TT(ns,ns), CC(ns), RR(ns,neps), fmat, fwt, ywt, gev
     double precision, intent(inout) :: loose

     !f2py depend(size(G0,1)) TT, C0, RR
     double complex, dimension(ns,ns) :: Q, Z, AA, BB, cG0,cG1
     double complex, dimension(ns) :: alpha, beta
     double complex :: cRR(ns, neps)
     integer, intent(out) :: eu(2)
     double complex, allocatable :: etawt(:,:), zwt(:,:)
     integer :: info, pin, n, i, ipiv(ns), ldzt, nstab

     double complex, allocatable :: Qstab(:,:), Qunstab(:,:)
     ! svd  stuff
     double complex, allocatable :: eta_u(:,:), eta_v(:,:), zwt_u(:,:), zwt_v(:,:)
     double precision, allocatable :: eta_s(:), zwt_s(:)
     double complex, allocatable :: zwt_u_tran(:,:), int_mat(:,:), int_mat2(:,:), tmat(:,:), vv2(:,:), cPI(:,:)

     integer :: ldvt, lmin, nbigev
     logical :: unique
     double precision :: norm

     !test stuff
     double precision, dimension(50,50) :: sreal, simag, treal, timag, qreal, qimag, zreal, zimag


     eu      = (/0, 0/)
     nbigev  = 0
     n       = size(G0,1)
     pin     = size(PI,2)
     zxz     = 0
     nunstab = 0
     stake   = 1.01d0
     fixdiv  = 1

     fmat = 0
     ywt = 0
     gev = 0
     fwt = 0

     TT = 0.0d0
     CC = 0.0d0
     RR = 0.0d0

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
        call zdscal(lmin, 1.0d0/eta_s(i), eta_v(i,:), 1)
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
     tmat = 0.0d0
     do i  = 1, nstab
        tmat(i,i) = 1.0d0
     end do

     ! int mat = deta\veta'*veta1*deta1*ueta1'
     call zgemm('n','c', lmin, nstab, pin, CPLX_ONE, eta_v, lmin, zwt, nstab, &
          CPLX_ZERO, int_mat, lmin)
     call zgemm('c','c', nstab, nunstab, lmin, CPLX_NEGONE, int_mat, lmin, &
          eta_u, nunstab, CPLX_ZERO, tmat(:, nstab+1:n), nstab)


     cG0 = dcmplx(0.0d0,0.0d0)
     cG0(1:(n-nunstab),:) = matmul(tmat,AA)

     do i = n-nunstab+1,n
        cG0(i,i) = dcmplx(1.0d0, 0.0d0)
     end do

     cG1 = dcmplx(0.0d0,0.0d0)
     cG1(1:(n-nunstab),:) = matmul(tmat,BB)

     call zgesv(n, n, cG0, n, ipiv, cG1, n, info)

     TT = real(matmul(matmul(Z, cG1), transpose(conjg(Z))));
     cRR = dcmplx(0.0d0)
     cRR(1:n-nunstab,:) = matmul(matmul(tmat,Q),PSI)

     cG0 = dcmplx(0.0d0,0.0d0)
     cG0(1:(n-nunstab),:) = matmul(tmat,AA)

     do i = n-nunstab+1,n
        cG0(i,i) = dcmplx(1.0d0, 0.0d0)
     end do

     call zgesv(n, size(RR, 2), cG0, n, ipiv, cRR, n, info)
     RR = real(matmul(Z,cRR))

     deallocate(etawt, eta_u, eta_s, eta_v)
     deallocate(zwt, zwt_u, zwt_s, zwt_v)!, zwt_u_tran)
     deallocate(int_mat, tmat, vv2, cPI, Qstab, Qunstab)

     !call mkl_free_buffers()
   end subroutine do_gensys

 subroutine compute_norm(d, norm, m, n)
   ! computes 2-norm of matrix d [m x n]
   double complex, intent(in) :: d(m, n)
   double precision, intent(out) :: norm
   integer, intent(in) :: m, n

   integer :: md, lwork, info
   double precision, allocatable :: rwork(:), norm_m(:)
   double complex, allocatable :: work(:)

   md = minval((/ m, n /),1)
   lwork = -1
   norm = 10.0d0
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

    double complex, intent(in) :: alpha, beta
    double precision :: A, B, divhat

    A = sqrt(real(alpha)**2 + aimag(alpha)**2)
    B = abs(real(beta))!sqrt(real(beta)**2.0 + aimag(beta)**2.0)

    if (A > 0.0d0) then
       divhat = B/A
       if (((fixdiv == 1) .and. (1.0d0 + verysmall < divhat)) .and. divhat < stake) then
          stake = (1.0d0 + divhat) / 2.0d0
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
    double complex, intent(out), dimension(n,n) :: q, z, aa, bb
    double complex, intent(out) :: alpha(n), beta(n)
    integer, intent(out) :: info

    integer :: n, sdim, lwork, i
    double complex, dimension(n, n) :: cplxa, cplxb
    double complex, allocatable :: work(:)
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
    double complex, intent(in) :: A(nrow, ncolumn)

    double complex, intent(out) :: U(nrow, nmin), V(nmin, ncolumn)
    double precision, intent(out) :: S(nmin)

    double complex :: AA(nrow, ncolumn)
    integer :: info, lwork
    double complex, allocatable :: work(:)
    double precision :: rwork(5*nmin)

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

  subroutine call_gensys(TT,CC,RR,fmat,fwt,ywt,gev,eu,loose, &
       G0, G1, C0, PSI, PI, DIV, ns, neps, neta)

    !use mkl95_precision, only: wp => dp

    integer, intent(in) :: ns, neps, neta

    double precision, intent(in) :: G0(ns,ns), G1(ns,ns), C0(ns), PSI(ns,neps), PI(ns,neta), DIV
    double precision, intent(out) :: TT(ns,ns), CC(ns), RR(ns,neps), fmat, fwt, ywt, gev, loose
    integer, intent(out) :: eu(2)


    double precision :: GG0(ns,ns), GG1(ns,ns), CC0(ns), PPSI(ns,neps), PPI(ns,neta), ddiv, lloose

    GG0  = G0
    GG1  = G1

    CC0  = C0
    PPSI = PSI
    PPI  = PI

    ddiv = DIV


    call do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, &
         size(G0,1), size(PSI,2), size(PI,2), &
       GG0, GG1, CC0, PPSI, PPI,ddiv)


  end subroutine call_gensys


end module gensys
