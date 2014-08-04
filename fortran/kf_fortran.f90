subroutine kalman_filter(y, TT, RR, QQ, DD, ZZ, HH, P0, ny, nobs, neps, ns, loglh)

  ! Evaluating the likelihood of LGSS via Kalman Filter.
  !--------------------------------------------------------------------------------
  integer, intent(in) :: ny, nobs, neps, ns
  double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps), DD(ny), ZZ(ny,ns), HH(ny,ny), P0(ns,ns)
  double precision, intent(out) :: loglh

  double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), Kt(ns,ny), QQRRp(neps,ns)
  double precision :: yhat(ny), nut(ny), Ft(ny,ny), iFt(ny,ny), detFt, M_PI
  integer :: t, info, t0

  double precision :: ZZP0(ny,ns), iFtnut(ny,ny), gain(ns), C(ns,ns), KtiFt(ns,ny), TTPt(ns,ns)
  double precision :: ONE, ZERO, NEG_ONE
  ! BLAS functions
  double precision :: ddot

  M_PI = 3.141592653589793d0
  ONE = 1.0d0
  ZERO = 0.0d0
  NEG_ONE = -ONE

  ! initialization
  At = 0.0d0
  call dgemm('n','t', neps, ns, neps, 1.0d0, QQ, neps, RR, ns, 0.0d0, QQRRp, neps)
  call dgemm('n','n', ns, ns, neps, 1.0d0, RR, ns, QQRRp, neps, 0.0d0, RQR, ns)

  !call dlyap(TT, RQR, Pt, ns, info)

  ! Pt = TT*Pt*TT' + RQR
  Pt = P0
  call dgemm('n','n', ns, ns, ns, 1.0d0, TT, ns, Pt, ns, 0.0d0, TTPt, ns)
  Pt = RQR
  call dgemm('n','t', ns, ns, ns, 1.0d0, TTPt, ns, TT, ns, 1.0d0, Pt, ns)

  loglh = 0.0d0
  t0 = 0

  do t = 1, nobs

     ! yhat = ZZ*At + DD
     call dcopy(ny, DD, 1, yhat, 1)
     call dgemv('n', ny, ns, ONE, ZZ, ny, At, 1, ONE, yhat, 1)

     ! nut = yt - yhat
     nut = y(:, t) - yhat

     ! Ft = ZZ*Pt*ZZ' + HH
     call dcopy(ny*ny, HH, 1, Ft, 1)
     call dsymm('r', 'l', ny, ns, ONE, Pt, ns, ZZ, ny, ZERO, ZZP0, ny)
     call dgemm('n', 't', ny, ny, ns, ONE, ZZP0, ny, ZZ, ny, ONE, Ft, ny)

     ! iFt = inv(Ft)
     call dcopy(ny*ny, Ft, 1, iFt, 1)
     call dpotrf('u', ny, iFt, ny, info)
     call dpotri('u', ny, iFt, ny, info)

     ! det(Ft)
     call determinant(Ft, ny, detFt)

     call dsymv('u', ny, ONE, iFt, ny, nut, 1, ZERO, iFtnut, 1)

     if (t > t0) then
        loglh = loglh - 0.5d0*ny*log(2*M_PI) - 0.5d0*log(detFt) &
             - 0.5d0*ddot(ny, nut, 1, iFtnut, 1)
     endif

     ! Kt = TT*Pt*ZZ'
     call dgemm('n','t', ns, ny, ns, ONE, TT, ns, ZZP0, ny, ZERO, Kt, ns)

     ! At = TT*At + Kt*iFt*nut'
     call dgemv('n', ns, ny, ONE, Kt, ns, iFtnut, 1, ZERO, gain, 1)
     call dgemv('n', ns, ns, ONE, TT, ns, At, 1, ONE, gain, 1)
     call dcopy(ns, gain, 1, At, 1)

     ! Pt = TT*Pt*TT' + RQR - Kt*iFt*Kt'
     call dgemm('n','n', ns, ns, ns, ONE, TT, ns, Pt, ns, ZERO, C, ns)
     call dsymm('r', 'u', ns, ns, ONE, Pt, ns, TT, ns, ZERO, C, ns)
     call dcopy(ns*ns, RQR, 1, Pt, 1)
     call dgemm('n', 't', ns, ns, ns, ONE, C, ns, TT, ns, ONE, Pt, ns)

     call dsymm('r', 'u', ns, ny, ONE, iFt, ny, Kt, ns, ZERO, KtiFt, ns)
     call dgemm('n', 't', ns, ns, ny, NEG_ONE, KtiFt, ns, Kt, ns, ONE, Pt, ns)

  end do


end subroutine kalman_filter

! subroutine kalman_everything(y, TT, RR, QQ, DD, ZZ, HH, P0, ny, nobs, neps, ns, Att, At1t, loglh)
!   !use mkl95_precision, only: wp => dp

!   ! Evaluating the likelihood of LGSS via Kalman Filter.
!   !--------------------------------------------------------------------------------
!   integer, intent(in) :: ny, nobs, neps, ns
!   double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps), DD(ny), ZZ(ny,ns), HH(ny,ny), P0(ns,ns)
!   double precision, intent(out) :: loglh(nobs)

!   double precision, intent(out) :: Att(nobs,ns), At1t(nobs,ns)

!  double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), Kt(ns,ny), QQRRp(neps,ns)
!   double precision :: yhat(ny), nut(ny), Ft(ny,ny), iFt(ny,ny), detFt, M_PI
!   integer :: t, info, t0

!   double precision :: ZZP0(ny,ns), iFtnut(ny,ny), gain(ns), C(ns,ns), KtiFt(ns,ny), TTPt(ns,ns)
!   double precision :: ONE, ZERO, NEG_ONE
!   ! BLAS functions
!   double precision :: ddot

!   M_PI = 3.141592653589793d0
!   ONE = 1.0d0
!   ZERO = 0.0d0
!   NEG_ONE = -ONE

!   ! initialization
!   At = 0.0d0
!   call dgemm('n','t', neps, ns, neps, 1.0d0, QQ, neps, RR, ns, 0.0d0, QQRRp, neps)
!   call dgemm('n','n', ns, ns, neps, 1.0d0, RR, ns, QQRRp, neps, 0.0d0, RQR, ns)

!   !call dlyap(TT, RQR, Pt, ns, info)

!   ! Pt = TT*Pt*TT' + RQR
!   Pt = P0
!   call dgemm('n','n', ns, ns, ns, 1.0d0, TT, ns, Pt, ns, 0.0d0, TTPt, ns)
!   Pt = RQR
!   call dgemm('n','t', ns, ns, ns, 1.0d0, TTPt, ns, TT, ns, 1.0d0, Pt, ns)

!   loglh = 0.0d0
!   t0 = 0

!   do t = 1, nobs

!      ! yhat = ZZ*At + DD
!      call dcopy(ny, DD, 1, yhat, 1)
!      call dgemv('n', ny, ns, ONE, ZZ, ny, At, 1, ONE, yhat, 1)

!      ! save Att
!      Att(t,:) = At

!      ! nut = yt - yhat
!      nut = y(:, t) - yhat

!      ! Ft = ZZ*Pt*ZZ' + HH
!      call dcopy(ny*ny, HH, 1, Ft, 1)
!      call dsymm('r', 'l', ny, ns, ONE, Pt, ns, ZZ, ny, ZERO, ZZP0, ny)
!      call dgemm('n', 't', ny, ny, ns, ONE, ZZP0, ny, ZZ, ny, ONE, Ft, ny)

!      ! iFt = inv(Ft)
!      call dcopy(ny*ny, Ft, 1, iFt, 1)
!      call dpotrf('u', ny, iFt, ny, info)
!      call dpotri('u', ny, iFt, ny, info)

!      ! det(Ft)
!      call determinant(Ft, ny, detFt)

!      call dsymv('u', ny, ONE, iFt, ny, nut, 1, ZERO, iFtnut, 1)

!      if (t > t0) then
!         loglh(t) = -0.5d0*ny*log(2*M_PI) - 0.5d0*log(detFt) &
!              - 0.5d0*ddot(ny, nut, 1, iFtnut, 1)
!      endif

!      ! Kt = TT*Pt*ZZ'
!      call dgemm('n','t', ns, ny, ns, ONE, TT, ns, ZZP0, ny, ZERO, Kt, ns)

!      ! At = TT*At + Kt*iFt*nut'
!      call dgemv('n', ns, ny, ONE, Kt, ns, iFtnut, 1, ZERO, gain, 1)
!      call dgemv('n', ns, ns, ONE, TT, ns, At, 1, ONE, gain, 1)
!      call dcopy(ns, gain, 1, At, 1)

!      ! Pt = TT*Pt*TT' + RQR - Kt*iFt*Kt'
!      call dgemm('n','n', ns, ns, ns, ONE, TT, ns, Pt, ns, ZERO, C, ns)
!      call dsymm('r', 'u', ns, ns, ONE, Pt, ns, TT, ns, ZERO, C, ns)
!      call dcopy(ns*ns, RQR, 1, Pt, 1)
!      call dgemm('n', 't', ns, ns, ns, ONE, C, ns, TT, ns, ONE, Pt, ns)

!      call dsymm('r', 'u', ns, ny, ONE, iFt, ny, Kt, ns, ZERO, KtiFt, ns)
!      call dgemm('n', 't', ns, ns, ny, NEG_ONE, KtiFt, ns, Kt, ns, ONE, Pt, ns)

!   end do

!   At1t = Att



! end subroutine kalman_everything

! subroutine particle_filter_1st_order(y, TT, RR, QQ, DD, ZZ, HH, P0,
!   npart, ny, nobs, neps, ns, hatloglh)
!   !use mkl95_precision, only: wp => dp

!   ! Evaluating the likelihood of LGSS via Kalman Filter.
!   !--------------------------------------------------------------------------------
!   integer, intent(in) :: ny, nobs, neps, ns
!   double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps), DD(ny), ZZ(ny,ns), HH(ny,ny), P0(ns,ns)
!   double precision, intent(out) :: loglh

!   double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), Kt(ns,ny), QQRRp(neps,ns)
!   double precision :: yhat(ny), nut(ny), Ft(ny,ny), iFt(ny,ny), detFt, M_PI
!   integer :: t, info, t0, j

!   double precision :: ZZP0(ny,ns), iFtnut(ny,ny), gain(ns), C(ns,ns), KtiFt(ns,ny), TTPt(ns,ns)
!   double precision :: ONE, ZERO, NEG_ONE
!   ! BLAS functions
!   double precision :: ddot

!   double precision :: St(ns,npart), Stold(ns,npart), eps(neps,npart), wt(npart)

!   M_PI = 3.141592653589793d0
!   ONE = 1.0d0
!   ZERO = 0.0d0
!   NEG_ONE = -ONE

!   ! initialization
!   At = 0.0d0
!   call dgemm('n','t', neps, ns, neps, 1.0d0, QQ, neps, RR, ns, 0.0d0, QQRRp, neps)
!   call dgemm('n','n', ns, ns, neps, 1.0d0, RR, ns, QQRRp, neps, 0.0d0, RQR, ns)

!   !call dlyap(TT, RQR, Pt, ns, info)

!   ! Pt = TT*Pt*TT' + RQR
!   Pt = P0
!   call dgemm('n','n', ns, ns, ns, 1.0d0, TT, ns, Pt, ns, 0.0d0, TTPt, ns)
!   Pt = RQR
!   call dgemm('n','t', ns, ns, ns, 1.0d0, TTPt, ns, TT, ns, 1.0d0, Pt, ns)

!   loglh = 0.0d0
!   t0 = 0

!   ! draw Stold


!   do t = 1, nobs

!      ! draw eps

!      ! St = TT St-1 + RR*chol(QQ)*eps t
!      ! note QQRRp is now chol(QQ)*RR'
!      call dgemm('n','n',ns,npart,neps,1.0d0,RRcQQ,ns,eps,neps,0.0d0,St,ns)
!      call dgemm('n','n',ns,npart,ns,1.0d0,TT,ns,Stold,ns,1.0d0,St,ns)


!      do j = 1, npart
!         ! yhat = ZZ*At + DD
!         call dcopy(ny, DD, 1, yhat, 1)
!         call dgemv('n', ny, ns, ONE, ZZ, ny, St(:,j), 1, ONE, yhat, 1)

!         ! nut = yt - yhat
!         nut = y(:, t) - yhat



!         ! Ft = ZZ*Pt*ZZ' + HH
!         call dcopy(ny*ny, HH, 1, Ft, 1)
!         !call dsymm('r', 'l', ny, ns, ONE, Pt, ns, ZZ, ny, ZERO, ZZP0, ny)
!         !call dgemm('n', 't', ny, ny, ns, ONE, ZZP0, ny, ZZ, ny, ONE, Ft, ny)

!         ! iFt = inv(Ft)
!         call dcopy(ny*ny, Ft, 1, iFt, 1)
!         call dpotrf('u', ny, iFt, ny, info)
!         call dpotri('u', ny, iFt, ny, info)

!         ! det(Ft)
!         call determinant(Ft, ny, detFt)

!         call dsymv('u', ny, ONE, iFt, ny, nut, 1, ZERO, iFtnut, 1)

!         lnpyi(j) = - 0.5d0*ny*log(2*M_PI) - 0.5d0*log(detFt) &
!              - 0.5d0*ddot(ny, nut, 1, iFtnut, 1)
!      endif

!   end do


!   ! update weights, calculate lnpy
!   wtsim = exp(lnpyi) * wtsim
!   if (t > t0) then
!      lnpy(t) =  log(sum(wtsim)/npart)
!   endif
!   wtsim = wtsim / sum(wtsim)

!   ESS = 1.d0 / sum(wtsim**2)

!   ! resample ...
!   if (ESS < npart / 2) then

!   end if

!   Stold = St

!   ! Kt = TT*Pt*ZZ'
!   call dgemm('n','t', ns, ny, ns, ONE, TT, ns, ZZP0, ny, ZERO, Kt, ns)

!   ! At = TT*At + Kt*iFt*nut'
!   call dgemv('n', ns, ny, ONE, Kt, ns, iFtnut, 1, ZERO, gain, 1)
!   call dgemv('n', ns, ns, ONE, TT, ns, At, 1, ONE, gain, 1)
!   call dcopy(ns, gain, 1, At, 1)

!   ! Pt = TT*Pt*TT' + RQR - Kt*iFt*Kt'
!   call dgemm('n','n', ns, ns, ns, ONE, TT, ns, Pt, ns, ZERO, C, ns)
!   call dsymm('r', 'u', ns, ns, ONE, Pt, ns, TT, ns, ZERO, C, ns)
!   call dcopy(ns*ns, RQR, 1, Pt, 1)
!   call dgemm('n', 't', ns, ns, ns, ONE, C, ns, TT, ns, ONE, Pt, ns)

!   call dsymm('r', 'u', ns, ny, ONE, iFt, ny, Kt, ns, ZERO, KtiFt, ns)
!   call dgemm('n', 't', ns, ns, ny, NEG_ONE, KtiFt, ns, Kt, ns, ONE, Pt, ns)

! end do


! end subroutine particle_filter_1st_order






subroutine determinant(matrix, r, det)


! Computes the determinant of symmetric square matrix, matrix (rank r).
integer, intent(in) :: r
double precision, intent(in) :: matrix(r,r)

double precision, intent(out) :: det

integer :: info, i, piv(r)
double precision :: matrix_copy(r, r)

call dcopy(r*r, matrix, 1, matrix_copy, 1)


call dpotrf('u', r, matrix_copy, r, info)

if (info .ne. 0) then
   !write(*,'(a,i4)') 'In determinant(), dgetrf returned error code ', info
   det = -10000.0d0
   return
end if

det = 1.0d0

do i = 1, r

   if (.true.) then !(piv(i) .ne. i) then
      det = det * matrix_copy(i, i) * matrix_copy(i,i)
   else
      det = det * matrix_copy(i, i)
   end if

end do
end subroutine determinant


! f2py  -c --opt=-O3 -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread -L/mq/home/m1eph00/lib -lslicot_sequential kf_fortran.f90  --compiler=intel --fcompiler=intelem -m kalman
