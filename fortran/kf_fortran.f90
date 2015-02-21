module filter
   use mkl_vsl_type
   use mkl_vsl

   implicit none

   integer, parameter :: wp = selected_real_kind(15)


   double precision, parameter :: ONE = 1.0d0, ZERO = 0.0d0, NEG_ONE = -1.0d0, M_PI = 3.141592653589793d0
   double precision, parameter :: really_small = 1e-10


 contains
   subroutine kalman_filter_missing_with_states(y, TT, RR, QQ, DD, ZZ, HH, t0, ny, nobs, neps, ns, &
        loglh, filtered_states, smooth_states) 
     ! Evaluating the log likelihood of LGSS via the Kalman Filter.
     !  Can handle missing data entered as `NaN.'
     !--------------------------------------------------------------------------------
     integer, intent(in) :: ny, nobs, neps, ns, t0
     double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps), DD(ny), ZZ(ny,ns), HH(ny,ny)
     double precision, intent(out) :: loglh(nobs), smooth_states(nobs,ns)
     double precision :: smooth_shocks(nobs,neps), smooth_vars(ns,ns,nobs)
     double precision, intent(out) :: filtered_states(nobs,ns)
     double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), QQRRp(neps,ns)
     double precision :: detFt
     integer :: t,i,j, info

     integer :: nmiss, ngood
     integer, allocatable :: oind(:)
     double precision, allocatable :: iFtnut(:),ZZPt(:,:), Ft(:,:), iFt(:,:), Kt(:,:), KtiFt(:,:), nut(:),yhat(:), L(:,:)
     double precision :: ZZP0(ny,ns), gain(ns), C(ns,ns),  TTPt(ns,ns), HHmiss(ny,ny), beta_(ns,ns,nobs), eye(ns,ns)

     double precision :: fcst_error(ny,nobs), L_mat(ny,ns,nobs), P_mat(ns,ns,nobs), cc(ns),cc_old(ns), psi_(ns,ns)
     ! BLAS functions
     double precision :: ddot

     ! initialization 
     At = 0.0_wp
     call dgemm('n','t', neps, ns, neps, 1.0_wp, QQ, neps, RR, ns, 0.0_wp, QQRRp, neps)
     call dgemm('n','n', ns, ns, neps, 1.0_wp, RR, ns, QQRRp, neps, 0.0_wp, RQR, ns)

     call dlyap(TT, RQR, Pt, ns, info)

     ! Pt = TT*Pt*TT' + RQR

     call dgemm('n','n', ns, ns, ns, 1.0_wp, TT, ns, Pt, ns, 0.0_wp, TTPt, ns)
     Pt = RQR
     call dgemm('n','t', ns, ns, ns, 1.0_wp, TTPt, ns, TT, ns, 1.0_wp, Pt, ns)

     eye = 0.0_wp
     do t = 1,ns
        eye(t,t) = 1.0_wp
     end do
     fcst_error = 0.0_wp  

     loglh = 0.0_wp
     P_mat = 0.0_wp
     L_mat = 0.0_wp
     do t = 1, nobs
        HHmiss = 0.0_wp

        nmiss = count(isnan(y(:,t)))

        ngood = ny - nmiss
        allocate(oind(ngood),iFtnut(ngood),ZZPt(ngood, ns),Ft(ngood, ngood), &
             iFt(ngood,ngood),Kt(ns,ngood), KtiFt(ns,ngood),nut(ngood),yhat(ngood),&
             L(ngood,ns))

        j = 1;
        do i = 1,ny
           if (isnan(y(i,t))) then 
           else
              oind(j) = i
              j = j + 1;
           end if
        end do


        smooth_states(t,:) = At
        P_mat(:,:,t) = Pt

        ! yhat = ZZ*At + DD
        call dcopy(ngood, DD(oind), 1, yhat, 1)
        call dgemv('n', ngood, ns, ONE, ZZ(oind,:), ngood, At, 1, ONE, yhat, 1)

        ! nut = yt - yhat
        nut = y(oind, t) - yhat
        fcst_error(1:ngood,t) = nut

        ! Ft = ZZ*Pt*ZZ' + HH
        call dcopy(ngood*ngood, HH(oind, oind), 1, Ft, 1)

        !call dsymm('r', 'l', ngood, ns, ONE, Pt, ns, ZZ(oind,:), ngood, ZERO, ZZPt, ngood)
        call dgemm('n','n', ngood, ns, ns, ONE, ZZ(oind, :), ngood, Pt, ns, ZERO, ZZPt, ngood)
        call dgemm('n', 't', ngood, ngood, ns, ONE, ZZPt, ngood, ZZ(oind,:), ngood, ONE, Ft, ngood)

        ! iFt = inv(Ft)
        call dcopy(ngood*ngood, Ft, 1, iFt, 1)
        call dpotrf('u', ngood, iFt, ngood, info)
        call dpotri('u', ngood, iFt, ngood, info)


        ! det(Ft)
        !detFt = determinant(Ft, ngood);
        call determinant(Ft, ngood, detFt)
        call dsymv('u', ngood, ONE, iFt, ngood, nut, 1, ZERO, iFtnut, 1)

        if (t > t0) then
           loglh(t) = - 0.5_wp*ngood*log(2*M_PI) - 0.5_wp*log(detFt) & 
                - 0.5_wp*ddot(ngood, nut, 1, iFtnut, 1)
        endif

        ! Kt = TT*Pt*ZZ'
        call dgemm('n','t', ns, ngood, ns, ONE, TT, ns, ZZPt, ngood, ZERO, Kt, ns)

        ! At = TT*At + Kt*iFt*nut'
        call dgemv('n', ns, ngood, ONE, Kt, ns, iFtnut, 1, ZERO, gain, 1)
        call dgemv('n', ns, ns, ONE, TT, ns, At, 1, ONE, gain, 1)
        call dcopy(ns, gain, 1, At, 1)

        filtered_states(t,:) = At

        ! Pt = TT*Pt*TT' + RQR - Kt*iFt*Kt'
        call dgemm('n','n', ns, ns, ns, ONE, TT, ns, Pt, ns, ZERO, C, ns)
        call dsymm('r', 'u', ns, ns, ONE, Pt, ns, TT, ns, ZERO, C, ns)
        call dcopy(ns*ns, RQR, 1, Pt, 1)
        call dgemm('n', 't', ns, ns, ns, ONE, C, ns, TT, ns, ONE, Pt, ns)

        call dsymm('r', 'u', ns, ngood, ONE, iFt, ngood, Kt, ns, ZERO, KtiFt, ns)
        call dgemm('n', 't', ns, ns, ngood, NEG_ONE, KtiFt, ns, Kt, ns, ONE, Pt, ns)

        ! from Hess's code
        !call dgemm('t','n',ns,ngood,ngood,ONE,ZZ(oind,:),ngood,iFt,ngood,0.0_wp,L,ns)
        call dsymm('l','u',ngood,ns,ONE,iFt,ngood,ZZ(oind,:),ngood,0.0_wp,L,ngood)
        L_mat(1:ngood,:,t) = L

        beta_(:,:,t) = eye
        !call dcopy(ns*ns,eye, 1, beta_(t,:,:),1)
        call dgemm('t','n',ns,ns,ngood,-1.0_wp,L,ngood,ZZPt,ngood,ONE,beta_(:,:,t),ns)

        ! if (t == nobs-1) then
        !    open(2,file='beta_pre.txt',action='write')
        !    do j = 1,ns
        !       write(2,'(200f32.8)') beta_(j,:,t)
        !    end do
        !    close(2)

        !    open(1,file='L2.txt',action='write')
        !    do j = 1,ngood
        !       write(1,'(200f32.8)') L(j,:)
        !    end do
        !    close(1)

        !    open(1,file='ZZPt.txt',action='write')
        !    do j = 1,ngood
        !       write(1,'(200f32.8)') ZZPt(j,:)
        !    end do
        !    close(1)

        ! end if

        if (t > 1) then
           call dgemm('n','t',ns,ns,ns,1.0_wp,beta_(:,:,t-1),ns,TT,ns,ZERO,TTPt,ns)
           !call dcopy(ns*ns, TTPt, 1, beta_(:,:,t-1), 1)
           beta_(:,:,t-1) = TTPt
        end if

        ! if (t == nobs) then
        !    open(2,file='beta_post.txt',action='write')
        !    do j = 1,ns
        !       write(2,'(200f32.8)') beta_(:,j,t-1)
        !    end do
        !    close(2)
        ! end if

        deallocate(oind)

        deallocate(Ft,iFt,iFtnut,Kt,KtiFt,ZZPt,nut,yhat,L)
     end do


     ! very rudimentary smoother
     cc = 0.0_wp
     cc_old = 0.0_wp

     psi_ = 0.0_wp

     do t=nobs,1,-1

        nmiss = count(isnan(y(:,t)))

        ngood = ny - nmiss
        allocate(oind(ngood))
        j = 1;
        do i = 1,ny
           if (isnan(y(i,t))) then 
           else
              oind(j) = i
              j = j + 1;
           end if
        end do

        ! psi = beta*psi_*beta' + L'ZZ;
        call dgemm('n','t',ns,ns,ns,ONE,psi_,ns,beta_(:,:,t),ns,ZERO,TTPt,ns)
        call dgemm('n','n',ns,ns,ns,ONE,beta_(:,:,t),ns,TTPt,ns,ZERO,psi_,ns)
        call dgemm('t','n',ns,ns,ngood,ONE,L_mat(1:ngood,:,t),ngood,ZZ(oind,:),ngood,ONE,psi_,ns)

        ! c = L_t*eta_t + beta_t*c
        ! a_tT = a_t + P_t*c
        cc_old = cc
        call dgemv('n',ns,ns,ONE,beta_(:,:,t),ns,cc_old,1,ZERO,cc,1)
        call dgemv('t',ngood,ns,ONE,L_mat(1:ngood,:,t),ngood,fcst_error(1:ngood,t),1,ONE,cc,1)
        call dgemv('n',ns,ns,ONE,P_mat(:,:,t),ns,cc,1,ONE,smooth_states(t,:),1)

        call dgemv('n',neps,ns,ONE,QQRRp,neps,cc,1,ONE,smooth_shocks(t,:),1)

        smooth_vars(:,:,t) = P_mat(:,:,t)
        call dgemm('n','n',ns,ns,ns,NEG_ONE,psi_,ns,P_mat(:,:,t),ns,ZERO,TTPt,ns)
        call dgemm('n','n',ns,ns,ns,ONE,P_mat(:,:,t),ns,psi_,ns,ONE,smooth_vars,ns)


        deallocate(oind)

     end do

     call mkl_free_buffers()
   end subroutine kalman_filter_missing_with_states

   subroutine part_filter(y, TT, RR, QQ, DD, ZZ, HH, t0, npart, filter_type, resampling,ny,nobs,neps,ns,incloglh,filtered_states,random_seed) 


     ! Evaluating the likelihood of LGSS via particle filter
     ! filter_type 
     !   0 -- bootstrap
     !   1 -- conditionally optimal 
     !   2 -- auxiliary particle filter
     !--------------------------------------------------------------------------------
     integer, intent(in) :: ny, nobs, neps, ns, t0, npart, filter_type, resampling, random_seed
     !f2py threadsafe

     double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps)
     double precision, intent(in) :: DD(ny), ZZ(ny,ns), HH(ny,ny)

     double precision, intent(out) :: incloglh(nobs)
     double precision, intent(out) :: filtered_states(nobs,ns)

     double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), Kt(ns,ny), QQRRp(neps,ns)
     double precision :: yhat(ny), nut(ny), Ft(ny,ny), iFt(ny,ny), detFt, TTPt(ns,ns)

     integer :: t, info, i, j

     double precision :: St(ns, npart), Stold(ns, npart)
     double precision :: wt(npart), wtold(npart), incwt(npart), loglh

     ! BLAS functions
     double precision :: ddot

     ! for singular value decomposition of state variance
     double precision :: U(ns,ns), s(ns), Vt(ns,ns), work(6*ns**2)
     double precision :: Stt(ns), ee(neps), Sto(ns), ZZSt(ny,npart),eta(ny,npart)
     integer :: iwork(8*ns)

     ! for g(s_t|s_{t-1}
     double precision :: iRQR(ns,ns), logdetRQR,RRcQQ(ns,neps)
     double precision :: rnd

     ! for lnpy
     double precision :: iHH(ny,ny), logdetHH, iHHnut(ny)
     double precision :: logdetFt, ZZP0(ny,ns), logdetPt, iPt(ns,ns), cPt(ns,ns), gain(ns)
     double precision :: iPteta(ns), Stbar(ns,npart), Stdiff(ns,npart)
     double precision :: lnp, lng, iFtZZP0(ny,ns)



     double precision :: ESS, Zt, uu(npart),cdf(npart)
     integer :: paraind(npart)
     ! fro random deviates
     type (VSL_STREAM_STATE) :: stream
     integer :: brng, seed, methodu, method, mstorage, time_array(8), errcode
     double precision :: eps(ns,npart), x(npart)


     !------------------------------------------------------------
     ! draw random deviates
     !------------------------------------------------------------
     call date_and_time(values=time_array)
     brng = VSL_BRNG_MT19937
     if (random_seed==0) then 
        seed = mod(sum(time_array),100000)
     else
        seed = random_seed
     end if
     errcode = vslnewstream(stream, brng, seed)

     method = VSL_METHOD_DGAUSSIANMV_BOXMULLER
     errcode = vdrnggaussian(method, stream, ns*npart, eps, 0.0d0, 1.0d0)

     methodu = VSL_METHOD_DUNIFORM_STD
     errcode = vdrnguniform(methodu, stream, nobs, x, 0.0d0, 1.0d0)


     !------------------------------------------------------------
     ! get initial variance of states
     !------------------------------------------------------------ 
     call dgemm('n','t', neps, ns, neps, 1.0d0, QQ, neps, RR, ns, 0.0d0, QQRRp, neps)
     call dgemm('n','n', ns, ns, neps, 1.0d0, RR, ns, QQRRp, neps, 0.0d0, RQR, ns)

     call dlyap(TT, RQR, Pt, ns, info)

     ! Pt = TT*Pt*TT' + RQR
     call dgemm('n','n', ns, ns, ns, 1.0d0, TT, ns, Pt, ns, 0.0d0, TTPt, ns)
     Pt = RQR
     call dgemm('n','t', ns, ns, ns, 1.0d0, TTPt, ns, TT, ns, 1.0d0, Pt, ns)
     !------------------------------------------------------------


     !------------------------------------------------------------
     ! draw initial particles s_0 ~ N(0,P0) L = sqrt(P0)
     !------------------------------------------------------------
     call dgesdd('A', ns, ns, Pt, ns, s, U, ns, Vt, ns, work, 6*ns**2, iwork, info)
     do i = 1,ns
        U(:,i) = U(:,i)*sqrt(s(i))
     end do
     call dgemm('n','n',ns,npart,ns,1.0d0,U,ns,eps(:,:),ns,0.0d0,Stold,ns)

     !------------------------------------------------------------


     !------------------------------------------------------------
     ! set up stuff for g(s_t|s_{t-1})
     !------------------------------------------------------------
     ! get sqrt(RRQQRR') (ASSUMES diagonal QQ)
     do i = 1,neps
        RRcQQ(:,i) = RR(:,i)*sqrt(QQ(i,i))
     end do

     ! get pseudo inverse (RR*QQ*RR')
     logdetRQR = 0.0d0
     call dgesdd('A', ns, ns, RQR, ns, s, U, ns, Vt, ns, work, 6*ns**2, iwork, info)
     do i = 1,ns
        if (s(i) > really_small) then
           U(:,i) = U(:,i)/s(i)
           logdetRQR = log(s(i)) + logdetRQR
        else
           U(:,i) = 0.0d0
        end if
     end do
     call dgemm('t','t',ns,ns,ns,1.0d0,Vt,ns,U,ns,0.0d0,iRQR,ns)

     !------------------------------------------------------------


     !------------------------------------------------------------
     ! set up stuff for conditionally optimal pf
     !------------------------------------------------------------
     if (filter_type == 1) then 
        ! SVD destroys (!?!?!) RQR on entry...
        call dgemm('n','t', neps, ns, neps, 1.0d0, QQ, neps, RR, ns, 0.0d0, QQRRp, neps)
        call dgemm('n','n', ns, ns, neps, 1.0d0, RR, ns, QQRRp, neps, 0.0d0, RQR, ns)

        Pt = RQR
        call dcopy(ny*ny, HH, 1, Ft, 1)
        call dsymm('r', 'l', ny, ns, ONE, Pt, ns, ZZ, ny, ZERO, ZZP0, ny)
        call dgemm('n', 't', ny, ny, ns, ONE, ZZP0, ny, ZZ, ny, ONE, Ft, ny)

        !do i = 1,ny
        !   write(*,'(100f7.3)') Ft(i,:)
        !end do

        ! iFt = inv(Ft)
        call dcopy(ny*ny, Ft, 1, iFt, 1)
        call dpotrf('u', ny, iFt, ny, info)
        call dpotri('u', ny, iFt, ny, info)

        ! det(Ft)
        !detFt = determinant(Ft, ny);
        call determinant(Ft, ny, detFt)
        logdetFt = log(detFt)

        ! Get the conditional variance for s_t|s_{t-1},y_t 
        call dsymm('l','u',ny,ns,1.0d0,iFt,ny,ZZP0,ny,0.0d0,iFtZZP0,ny)
        call dgemm('t','n',ns,ns,ny,-1.0d0,ZZP0,ny,iFtZZP0,ny,1.0d0,Pt,ns)

        logdetPt = 0.0d0
        call dgesdd('A', ns, ns, Pt, ns, s, U, ns, Vt, ns, work, 6*ns**2, iwork, info)
        do i = 1,ns
           cPt(:,i) = U(:,i)*sqrt(s(i))
           if (s(i) > really_small) then
              U(:,i) = U(:,i)/s(i)
              logdetPt = log(s(i)) + logdetPt
           else
              U(:,i) = 0.0d0
           end if

        end do
        call dgemm('t','t',ns,ns,ns,1.0d0,Vt,ns,U,ns,0.0d0,iPt,ns)

     end if


     !------------------------------------------------------------
     ! set up stuff for lnpy
     !------------------------------------------------------------
     ! ASSUMES diagonal HH
     iHH = 0.0d0
     logdetHH = 0.0d0
     do i = 1,ny
        iHH(i,i) = 1.0d0 / HH(i,i)
        logdetHH = log(HH(i,i)) + logdetHH
     end do
     !------------------------------------------------------------

     wt = 1.0d0
     incloglh = 0.0d0

     filtered_states = 0.0d0

     do t = 1,nobs
        errcode = vdrnggaussian(method, stream, ns*npart, eps, 0.0d0, 1.0d0)
        errcode = vdrnguniform(methodu, stream, npart, x, 0.0d0, 1.0d0)

        yhat = y(:,t) - DD

        !------------------------------------------------------------
        ! draw St| St[,y_t]
        !------------------------------------------------------------
        if (filter_type == 0) then
           call dgemm('n','n',ns,npart,neps,1.0d0,RRcQQ,ns,eps(1:neps,:),neps,0.0d0,St,ns)
           call dgemm('n','n',ns,npart,ns,1.0d0,TT,ns,Stold,ns,1.0d0,St,ns)
           call dgemm('n','n',ny,npart,ns,1.0d0,ZZ,ny,St,ns,0.0d0,ZZSt,ny)
        elseif (filter_type == 1) then
           Stold(:,1) = 1.0d0
           call dgemm('n','n',ns,npart,ns,1.0d0,TT,ns,Stold,ns,0.0d0,St,ns)
           call dgemm('n','n',ny,npart,ns,1.0d0,ZZ,ny,St,ns,0.0d0,eta,ny)

           Stbar = St
           do i = 1,npart
              eta(:,i) = yhat - eta(:,i)
           end do
           call dgemm('t','n', ns,npart,ny,1.0d0,iFtZZP0,ny,eta,ny,1.0d0,Stbar,ns)
           St = Stbar 
           call dgemm('n','n',ns,npart,ns,1.0d0,cPt,ns,eps,ns,1.0d0,St,ns)
           call dgemm('n','n',ny,npart,ns,1.0d0,ZZ,ny,St,ns,0.0d0,ZZSt,ny)

           Stdiff = St
           call dgemm('n','n',ns,npart,ns,-1.0d0,TT,ns,Stold,ns,1.0d0,Stdiff,ns)
        end if


        !ffdsf$OMP PARALLEL DO PRIVATE(i,ee,Sto,nut,iHHnut,Stt) SHARED(incwt,logdetHH,ny,St,ZZ,TT,RRcQQ,eps,yhat,t,ns,neps,npart,iHH,Stold,ZZSt)
        !------------------------------------------------------------
        ! calculate lnpyi 
        !------------------------------------------------------------
        do i = 1,npart
           ! Stt = 0.0d0
           ! Sto = Stold(:,i)

           ! ee = eps(1:neps,i)
           ! call dgemv('n',ns,neps,1.0d0,RRcQQ,ns,ee,1,0.0d0,Stt,1)  
           ! call dgemv('n',ns,ns,1.0d0,TT,ns,Sto,1,1.0d0,Stt,1)
           ! Stt = Stt + matmul(TT,Sto)

           ! call dgemv('n',ny,ns,-1.0d0,ZZ,ny,Stt,1,1.0d0,nut,1)
           ! nut = yhat - matmul(ZZ,Stt)
           nut = yhat - ZZSt(:,i)

           call dsymv('u', ny, ONE, iHH, ny, nut, 1, ZERO, iHHnut, 1)
           incwt(i) = -0.5d0*ny*log(2*M_PI) - 0.5d0*logdetHH &
                -0.5d0*ddot(ny,nut,1,iHHnut,1)

           if (filter_type == 1) then 

              gain = St(:,i) - Stbar(:,i)

              !call dsymv('u', ns, ONE, iPt, ns, gain, 1, ZERO, iPteta, 1)
              call dgemv('n', ns, ns, ONE, iPt, ns, gain, 1, ZERO, iPteta, 1)

              ! lng
              lng = -0.5d0*neps*log(2*M_PI) - 0.5d0*logdetPt &
                   -0.5d0*ddot(ns,gain,1,iPteta,1)

              gain = Stdiff(:,i) 

              !call dsymv('u', ns, ONE, iRQR, ns, gain, 1, ZERO, iPteta, 1)
              call dgemv('n', ns, ns, ONE, iRQR, ns, gain, 1, ZERO, iPteta, 1)

              ! lng
              lnp = -0.5d0*neps*log(2*M_PI) - 0.5d0*logdetRQR &
                   -0.5d0*ddot(ns,gain,1,iPteta,1)
              incwt(i) = incwt(i) + lnp - lng

           end if

        end do


        !------------------------------------------------------------
        ! calculate new weights
        !------------------------------------------------------------
        wt = exp(incwt) * wt
        Zt = sum(wt)/(1.0d0*npart)

        incloglh(t) = log(Zt)

        wt = wt / Zt

        !------------------------------------------------------------
        ! save states
        !------------------------------------------------------------
        do i = 1,npart
           filtered_states(t,:) = filtered_states(t,:) + St(:,i)*wt(i)/(1.0d0*npart)
        end do

        !------------------------------------------------------------
        ! resample, if necessary
        !------------------------------------------------------------
        ESS = npart**2 / sum(wt**2)


        if (ESS < npart/2) then 
           if (resampling==0) then 
              call sys_resampling(npart, wt/sum(wt), x(t), paraind)
              Stold = St(:,paraind)
           elseif (resampling==1) then
              call mult_resampling(npart, wt/sum(wt), x, paraind)
              Stold = St(:,paraind)
           elseif (resampling == 2) then 
              !wt = 1.0d0
              wtold = wt/sum(wt)
              cdf(1) = wtold(1)
              do i=2,npart
                 cdf(i) = cdf(i-1) + wtold(i)
              end do

              ! draw a starting point
              !call random_number(rnd)
              rnd = x(1)
              uu = ( rnd -1.0d0 + real( (/ (i, i=1,npart) /) ,8) ) / real(npart,8)
              ! start at the bottom of the CDF
              j=1
              do i=1,npart
                 ! move along the CDF
                 do while (uu(i)>cdf(j))
                    j=j+1
                 end do
                 ! shuffling
                 Stold(:,i) = St(:,j)
              end do
           end if

           wt = 1.0d0
        else
           Stold = St
        end if



     end do

     loglh = sum(incloglh)


   end subroutine part_filter


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

  subroutine kalman_filter_hstep_quasilik(y, TT, RR, QQ, DD, ZZ, HH, P0, h, ny, nobs, neps, ns, loglh)

    ! Evaluating the likelihood of LGSS via Kalman Filter.
    !--------------------------------------------------------------------------------
    integer, intent(in) :: ny, nobs, neps, ns, h
    double precision, intent(in) :: y(ny,nobs), TT(ns,ns), RR(ns,neps), QQ(neps,neps)
    double precision, intent(in) :: DD(ny), ZZ(ny,ns), HH(ny,ny), P0(ns,ns)
    double precision, intent(out) :: loglh

    double precision :: At(ns), Pt(ns,ns), RQR(ns,ns), Kt(ns,ny), QQRRp(neps,ns)
    double precision :: yhat(ny), nut(ny), Ft(ny,ny), iFt(ny,ny), detFt, M_PI
    integer :: t, info, t0

    double precision :: ZZP0(ny,ns), iFtnut(ny,ny), gain(ns), C(ns,ns), KtiFt(ns,ny), TTPt(ns,ns)
    double precision :: ONE, ZERO, NEG_ONE

    ! for h step average 
    double precision :: TTn(ns,ns), TTn1(ns,ns), TTn_sum(ns,ns), ZZTTn(ny,ns), temp(ns,ns)
    double precision :: Fth(ny,ny), iFth(ny,ny), ZZTTnPt(ny,ns), detFth, iFthnut(ny,ny), nuth(ny)
    double precision :: RQR_sum(ns,ns), RQRtemp(ns,ns),ZZRQR_sum(ny,ns),ZZRQRsum_HH(ny,ny)
    integer :: i

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

    !------------------------------------------------------------
    ! set up (I + TT ... TT^(h-1))
    !------------------------------------------------------------
    TTn_sum = 0.0d0
    do i = 1,ns
       TTn_sum(i,i) = 1.0d0
    end do

    TTn1 = TTn_sum
    RQR_sum = 0.0d0
    do i = 2,h
       call dgemm('n','n',ns,ns,ns,1.0d0,TTn1,ns,RQR,ns,0.0d0,RQRtemp,ns)
       call dgemm('n','t',ns,ns,ns,1.0d0,RQRtemp,ns,TTn1,ns,1.0d0,RQR_sum,ns)


       call dgemm('n','n',ns,ns,ns,1.0d0,TT,ns,TTn1,ns,0.0d0,TTn,ns)
       TTn_sum = TTn_sum + TTn
       TTn1 = TTn
    end do


    call dgemm('n','n',ny,ns,ns,1.0d0,ZZ,ny,TTn_sum,ns,0.0d0,ZZTTn,ny)

    call dgemm('n','n',ny,ns,ns,1.0d0,ZZ,ny,RQR_sum,ns,0.0d0,ZZRQR_sum,ny)
    ZZRQRsum_HH = (1.0d0 * h) * HH
    call dgemm('n', 't', ny, ny, ns, ONE, ZZRQR_sum, ny, ZZ, ny, ONE, ZZRQRsum_HH, ny)
    !------------------------------------------------------------

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

       if ((t > t0) .and. (t <= nobs - h + 1)) then

          call dcopy(ny, DD, 1, yhat, 1)
          call dgemv('n', ny, ns, (1.0d0/(1.0d0 * h)), ZZTTn, ny, At, 1, ONE, yhat, 1)

          nuth = sum(y(:,t:t+h-1),dim=2)/(1.0d0*h) - yhat

          call dcopy(ny*ny, ZZRQRsum_HH, 1, Fth, 1)
          call dsymm('r', 'l', ny, ns, ONE, Pt, ns, ZZTTn, ny, ZERO, ZZTTnPt, ny)
          call dgemm('n', 't', ny, ny, ns, ONE, ZZTTnPt, ny, ZZTTn, ny, ONE, Fth, ny)
          Fth = Fth/(1.0d0 * (h**2))


          call dcopy(ny*ny, Fth, 1, iFth, 1)
          call dpotrf('u', ny, iFth, ny, info)
          call dpotri('u', ny, iFth, ny, info)

          ! det(Ft)
          call determinant(Fth, ny, detFth)
          call dsymv('u', ny, ONE, iFth, ny, nuth, 1, ZERO, iFthnut, 1)

          loglh = loglh - 0.5d0*ny*log(2*M_PI) - 0.5d0*log(detFth) &
               - 0.5d0*ddot(ny, nuth, 1, iFthnut, 1)
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


  end subroutine kalman_filter_hstep_quasilik



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

subroutine dlyap(TT, RQR, P0, ns, info)
  ! Computes the solution to the discrete Lyapunov equation,
  !      P0 = TT*P0*TT' + RQR
  ! where (inputs) TT, RQR and (output) P0 are ns x ns (real) matrices. 
  !--------------------------------------------------------------------------------
  !use mkl95_precision, only: wp => dp

  integer, intent(in) :: ns
  double precision, intent(in) :: TT(ns,ns), RQR(ns,ns)

  integer, intent(out) :: info
  double precision, intent(out) :: P0(ns,ns)

  ! for slicot
  double precision :: scale, U(ns,ns), UH(ns, ns), rcond, ferr, wr(ns), wi(ns), dwork(14*ns*ns*ns), sepd
  integer :: iwork(ns*ns), ldwork

  integer :: t

  UH = TT
  P0 = -1.0d0*RQR


  call sb03md('D','X', 'N', 'T', ns, UH, ns, U, ns, P0, ns, &
       scale, sepd, ferr, wr, wi, iwork, dwork, 14*ns*ns*ns, info)
 
  !if (ferr > 0.000001d0) call dlyap_symm(TT, RQR, P0, ns, info)
  if (info .ne. 0) then
     print*,'SB03MD failed. (info = ', info, ')'
     P0 = 0.0d0
     info = 1
     do t = 1,ns
	P0(t,t)=1.0d0
     end do

     return
  else
     
     !	     P0 = 0.5d0*P0 + 0.5d0*transpose(P0)
     info = 0

  end if


  
end subroutine dlyap

  subroutine mult_resampling(npart, wtsim, randu, paraind)

    integer, intent(in) :: npart

    real(8), intent(in) :: wtsim(npart), randu(npart)
    integer, intent(out) :: paraind(npart)

    real(8) :: cumsum(npart), u
    integer :: i, j


    do i = 1, npart
       cumsum(i) = sum(wtsim(1:i))
    end do

    do i = 1, npart

       u = randu(i)

       j = 1
       do 
          if (u < cumsum(j)) exit
          
          j = j + 1
       end do

       paraind(i) = j

    end do

  end subroutine mult_resampling

  subroutine sys_resampling(npart, wtsim, randu, paraind)
    
    integer, intent(in) :: npart
    
    real(8), intent(in) :: wtsim(npart), randu
    integer, intent(out) :: paraind(npart)

    integer :: cweights(npart), m(npart), k, cs

    cweights(1) = floor(npart*wtsim(1) + randu)
    m(1) = cweights(1)

    paraind = 0
    paraind(1:m(1)) = 1

    cs = m(1)
    do k = 2, npart
       cweights(k) = floor(sum(npart*wtsim(1:k))  + randu)

       if (k == npart) then
          cweights(npart) = npart
       end if
       m(k) = cweights(k) - cweights(k-1)
       paraind(cs+1:cs+m(k)) = k

       cs = cs + m(k)
       

    end do

    
    ! do k = 2, npart
    !   m(k) = cweights(k) - cweights(k-1)p
    ! end do

    ! paraind = 0
    ! paraind(1:m(1)) = 1

    ! cs = m(1)

    ! do k = 2, npart
    !    paraind(cs+1:cs+m(k)) = k

    !    cs = cs + m(k)

    ! enddo

  end subroutine sys_resampling


end module filter
! f2py  -c --opt=-O3 -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread -L/mq/home/m1eph00/lib -lslicot_sequential kf_fortran.f90  --compiler=intel --fcompiler=intelem -m kalman
