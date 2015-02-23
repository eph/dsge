!!! meyer_gohde.f90 --- 
!! 
!! Description: Module to implement 
!!    A. Meyer-Gohde: "Linear Rational-Expectations Models
!!        With Lagged Expectations: A Synthetic Method"
!! 
!! Author: Ed Herbst [edward.p.herbst@frb.gov]
!! Last-Updated: 06/11/14
!! 
include 'mkl_pardiso.f90'

module mg
  use mkl_pardiso
  


  double precision, parameter :: verysmall = 0.0d0!000001d0
  integer :: nunstab = 0
  integer :: zxz = 0 
  integer :: fixdiv = 1
  double precision :: stake = 1.00d0


contains

  logical function delctg(alpha, beta)

    double complex, intent(in) :: alpha, beta
    double precision :: A, B, divhat

    A = sqrt(real(alpha)**2 + aimag(alpha)**2)
    B = sqrt(real(beta)**2 + aimag(beta)**2)

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
       delctg = .true.
    else
       delctg = .false.
    end if

  end function delctg

  subroutine qz(a, b, aa, bb, q, z, alpha, beta, info, n)

    integer, intent(in) :: n
    double precision, intent(in) :: a(n,n), b(n,n)
    double complex, intent(out), dimension(n,n) :: q, z, aa, bb
    double complex, intent(out) :: alpha(n), beta(n)
    integer, intent(out) :: info

    integer :: sdim, lwork, i
    double complex, dimension(n, n) :: cplxa, cplxb
    double complex, allocatable :: work(:)
    double precision, dimension(8*n) :: rwork
    logical, dimension(n) :: bwork

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
    nunstab = 0
    allocate(work(lwork))
    call zgges('V','V','S', delctg, n, cplxa, n, cplxb, n, sdim, alpha, beta, q, &
         n, z, n, work, lwork, rwork, bwork, info)

    deallocate(work)

    nunstab = n - sdim
    aa = cplxa
    bb = cplxb


  end subroutine qz



  subroutine qz_solve(Ai,Bi,Ci,Fi,Gi,N,RC,ralpha,rbeta,neq,neps)

    integer, intent(in) :: neq,neps

    double precision, intent(in) :: Ai(neq,neq), Bi(neq,neq), Ci(neq,neq), Fi(neq,neps), Gi(neq,neps), N(neps,neps)

    integer, intent(out) :: RC

    double precision, intent(out) :: ralpha(neq,neq), rbeta(neq,neps)

    double complex :: TTS(2*neq,2*neq),SSS(2*neq,2*neq),QS(2*neq,2*neq),ZS(2*neq,2*neq)
    double complex :: alpha(2*neq), beta(2*neq)
    double precision :: AA(neq*2,neq*2), BB(neq*2,neq*2), CC(neq*2,neps)

    integer :: i, j,info,ipiv(neq), rank

    double complex :: QC(neq,neps), r_tran(neps), C_ZERO, C_ONE, Gamma(neq,neps), Nc(neps,neps), Ni(neps,neps), QSt(neq,2*neq)

    double complex :: alpha_zs(neq,neq), beta_zs(neq,neps), izs(neq,neq), work(neq*10), zs_temp(neq,neq), ZSupp(neq,neq)

    double precision :: read_in(2*neq), read_inc(2*neq), sing_val(neq), rwork(5*neq)

    C_ZERO = dcmplx(0.0d0,0.0d0)
    C_ONE  = dcmplx(1.0d0,0.0d0)

    AA = 0.0d0
    BB = 0.0d0
    CC = 0.0d0

    AA(1:neq,neq+1:2*neq) = -Ai
    BB(1:neq,1:neq) = Ci
    BB(1:neq,neq+1:2*neq) = Bi
    CC(1:neq,:) = Gi

    call dgemm('n','n',neq,neps,neps,1.0d0,Fi,neq,N,neps,1.0d0,CC(1:neq,:),neq)

    do i = 1,neq
       AA(neq+i,i) = 1.0d0
       BB(neq+i,neq+i) = 1.0d0
    end do


    call qz(BB,AA,TTS,SSS,QS,ZS,alpha,beta,info,2*neq)





    if (nunstab==neq) then !existence & uniqueness
       RC = 0
    elseif (nunstab > neq) then !unstable
       print*,'System is unstable'
       RC = 1 
       return
    else !indeterminant
       print*,'System is indeterminant'
       RC = 2
       return
    end if

    ! find whether eigenvalues are 'translatable' to missing initial conditions
    ZSupp = ZS(1:neq,1:neq)
    rank = 0
    call zgesvd('n','n',neq,neq,ZSupp,neq,sing_val,ZSupp,neq,ZSupp,neq,work,10*neq,rwork,info)

    do i = 1,neq
       if (sing_val(i) > 1e-13) then !should be written a bit more carefully
          rank = rank + 1
       end if
    end do

    if (rank < neq) then
       RC = 3
       print*,'Untranslatable initial conditions.'
       return
    end if



    Gamma = C_ZERO
    QC = C_ZERO
    QSt = QS(neq+1:2*neq,:)
    !call zgemm('n','n', neq, neps, 2*neq, C_ONE, QSt, neq, CC, 2*neq, C_ZERO, QC, neq)
    QC = matmul(QSt,CC)


    Nc = dcmplx(N,0.0d0)

    do j = 1,neq
       r_tran = QC(neq+1-j,:)
       do i = neq+2-j,neq
          r_tran = r_tran + TTS(2*neq+1-j,i+neq)*Gamma(i,:) 
          call zgemv('c',neps,neps,-SSS(2*neq+1-j,i+neq),Nc,neps,Gamma(i,:),1,C_ONE,r_tran,1)
       end do
       Ni = SSS(2*neq+1-j,2*neq+1-j)*N
       do i = 1,neps
          Ni(i,i) = Ni(i,i) - TTS(2*neq+1-j,2*neq+1-j)
       end do
       call zgetrf(neps,neps,Ni,neps,ipiv,info)
       call zgetrs('c',neps,1,Ni,neps,ipiv,r_tran,neps,info)
       Gamma(neq+1-j,1:neps) = r_tran
    end do

    izs = ZS(1:neq,1:neq)
    call zgetrf(neq,neq,izs,neq,ipiv,info)
    call zgetri(neq,izs,neq,ipiv,work,neq*10,info)

    alpha_zs = matmul(ZS(neq+1:2*neq,1:neq),izs)
    !call zgemm('n','n',neq,neq,neq,C_ONE,ZS(neq+1:2*neq,1:neq),neq,izs,neq,C_ZERO,alpha_zs,neq)

    izs = ZS(neq+1:2*neq,neq+1:2*neq) - alpha_zs

    izs = matmul(alpha_zs,ZS(1:neq,neq+1:2*neq))
    !call zgemm('n','n',neq,neq,neq,C_ONE,alpha_zs,neq,ZS(1:neq,neq+1:2*neq),neq,C_ZERO,izs,neq)
    zs_temp = ZS(neq+1:2*neq,neq+1:2*neq) - izs

    beta_zs = matmul(zs_temp,Gamma)
    !call zgemm('n','n',neq,neps,neq,C_ONE,zs_temp,neq,Gamma,neq,C_ZERO,beta_zs,neq)

    rbeta = real(beta_zs)
    ralpha = real(alpha_zs)
    if (any(isnan(rbeta)) .or. any(isnan(ralpha))) then
       print*,BB,AA
       stop
    end if

    call mkl_free_buffers() 

  end subroutine qz_solve

  subroutine find_max_it(pyAj,pyBj,pyCj,pyFj,pyGj,neq,neps,max_it)

    integer, intent(in) :: neq,neps
    integer, intent(out) :: max_it
    

    integer :: j

    double precision :: A(neq,neq),B(neq,neq),C(neq,neq),F(neq,neps),G(neq,neps)
    double precision :: Amax, Bmax,Cmax,Fmax,Gmax,all_max
    external ::   pyAj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyAj(ii,AAj,neq)


    external ::   pyBj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyBj(ii,AAj,neq)

    external ::   pyCj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyCj(ii,AAj,neq)

    external ::   pyFj
    !f2py integer ii
    !f2py double precision AFj(neq,neps)
    !f2py intent(out) AFj(neq,neps)
    !f2py call pyFj(ii,AFj,neq,neps)

    external ::   pyGj
    !f2py integer ii
    !f2py double precision AFj(neq,neps)
    !f2py intent(out) AFj(neq,neps)
    !f2py call pyGj(ii,AFj,neq,neps)



    do j = 1,5000


       call pyAj(j,A,neq)
       call pyBj(j,B,neq)
       call pyCj(j,C,neq)
       call pyFj(j,F,neq,neps)
       call pyGj(j,G,neq,neps)

       Amax = maxval(abs(A))
       Bmax = maxval(abs(B))
       Cmax = maxval(abs(C))
       Fmax = maxval(abs(F))
       Gmax = maxval(abs(G))

       all_max = max(Amax,Bmax,Cmax,Fmax,Gmax)

       if (all_max < convergence_threshold) then
          max_it = j
          return
       end if

    end do

    max_it = j

    return

  end subroutine find_max_it

  subroutine solve_ma_alt(iiA,iiB,iiC,iiF,iiG,iiN,pyAj,pyBj,pyCj,pyFj,pyGj,iiAinf,iiBinf,iiCinf,iiFinf,iiGinf,&
       max_it, MA_VECTOR, ALPHA, BETA, RC, neq,neps)



    integer, intent(in) :: max_it
    integer, intent(in) :: neq, neps
    double precision, intent(in):: iiA(neq,neq),iiB(neq,neq),iiC(neq,neq),iiF(neq,neps),iiG(neq,neps),iiN(neps,neps)
    double precision, intent(in):: iiAinf(neq,neq),iiBinf(neq,neq),iiCinf(neq,neq),iiFinf(neq,neps),iiGinf(neq,neps)
    
    double precision, intent(out) :: MA_VECTOR((max_it+1)*neq,neps)
    double precision, intent(out) :: ALPHA(neq,neq),BETA(neq,neps)
    integer, intent(out) :: RC



    double precision :: RHS((max_it+1)*neq,neps)
    double precision, allocatable :: RHS2(:), MA_VECTOR2(:)
    double precision :: XXX((max_it-1)*neq*neq*3+4*neq*neq)
    integer  :: III((max_it-1)*neq*neq*3+4*neq*neq),JJJ((max_it-1)*neq*neq*3+4*neq*neq)


    double precision :: EYE(neq,neq)
    double precision :: A(neq,neq),B(neq,neq),C(neq,neq),F(neq,neps),G(neq,neps), N(neps,neps)
    double precision :: Ainf(neq,neq),Binf(neq,neq),Cinf(neq,neq),Finf(neq,neps),Ginf(neq,neps)


    double precision :: LHS((max_it-1)*neq*neq*3+4*neq*neq)
    integer :: ja((max_it-1)*neq*neq*3+4*neq*neq), ia((max_it+1)*neq+1), job(6), nnz,info

    integer :: row_vec(neq)
    integer :: gap, i, g0


    character :: matdescra(6)

    integer :: maxfct, mnum, mtype, phase, iparm(64), error, msglvl

    integer :: perm((max_it+1)*neq), nrhs, idum(1), error1

    double precision :: Aj(neq,neq),Bj(neq,neq),Cj(neq,neq),Fj(neq,neps),Gj(neq,neps)    

    external ::   pyAj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyAj(ii,AAj,neq)


    external ::   pyBj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyBj(ii,AAj,neq)

    external ::   pyCj
    !f2py integer ii
    !f2py double precision AAj(neq,neq)
    !f2py intent(out) AAj(neq,neq)
    !f2py call pyCj(ii,AAj,neq)

    external ::   pyFj
    !f2py integer ii
    !f2py double precision AFj(neq,neps)
    !f2py intent(out) AFj(neq,neps)
    !f2py call pyFj(ii,AFj,neq,neps)

    external ::   pyGj
    !f2py integer ii
    !f2py double precision AFj(neq,neps)
    !f2py intent(out) AFj(neq,neps)
    !f2py call pyGj(ii,AFj,neq,neps)


    TYPE(MKL_PARDISO_HANDLE), allocatable :: pt(:)
    double precision :: ddum(1)
            
  
    MA_VECTOR = 0.0d0
    ALPHA = 0.0d0
    BETA = 0.0d0
    
    A = iiA 
    B = iiB
    C = iiC
    F = iiF
    G = iiG
    N = iiN

    Ainf = iiAinf
    Binf = iiBinf
    Cinf = iiCinf
    Finf = iiFinf
    Ginf = iiGinf
    

    !call zero_matrices(A,B,C,F,G,para,neq,neps,npara)
    !call N_matrix(N, para, neps)
    do i = 1,neq
       row_vec(i) = i
    end do

    gap = 1;
    g0 = neq**2

    ! A0 B0
    do gap = 1,neq

       XXX(neq*(gap-1)+1:neq*gap) = B(:,gap)
       III(neq*(gap-1)+1:neq*gap) = row_vec
       JJJ(neq*(gap-1)+1:neq*gap) = gap


       XXX(g0+ neq*(gap-1)+1:neq*gap+g0) = A(:,gap)
       III(g0+ neq*(gap-1)+1:neq*gap+g0) = row_vec
       JJJ(g0+ neq*(gap-1)+1:neq*gap+g0) = gap + neq


    end do

    RHS(1:neq,:) = G
    call dgemm('n','n',neq,neps,neps,-1.0d0,F,neq,N,neps,-1.0d0,RHS(1:neq,:),neq)

    g0 = 2*neq**2

    do i = 1, max_it - 1

       !call j_matrices(Aj,Bj,Cj,Fj,Gj,i,para,neq,neps,npara)

       call pyAj(i,Aj,neq)
       call pyBj(i,Bj,neq)
       call pyCj(i,Cj,neq)
       call pyFj(i,Fj,neq,neps)
       call pyGj(i,Gj,neq,neps)

       C = C + Cj
       B = B + Bj
       A = A + Aj

       do gap = 1,neq
          XXX(g0+neq*(gap-1)+1:neq*gap+g0) = C(:,gap) 
          III(g0+neq*(gap-1)+1:neq*gap+g0) = row_vec + i*neq
          JJJ(g0+neq*(gap-1)+1:neq*gap+g0) = (i-1)*neq + gap
       end do
       g0 = g0 + neq**2


       do gap = 1,neq
          XXX(g0+neq*(gap-1)+1:neq*gap+g0) = B(:,gap) 
          III(g0+neq*(gap-1)+1:neq*gap+g0) = row_vec + i*neq
          JJJ(g0+neq*(gap-1)+1:neq*gap+g0) = i*neq + gap
       end do
       g0 = g0 + neq**2


       do gap = 1,neq
          XXX(g0+neq*(gap-1)+1:neq*gap+g0) = A(:,gap) 
          III(g0+neq*(gap-1)+1:neq*gap+g0) = row_vec + i*neq
          JJJ(g0+neq*(gap-1)+1:neq*gap+g0) = (i+1)*neq + gap
       end do
       g0 = g0 + neq**2

       call dgemm('n','n',neq,neps,neps,1.0d0,G,neq,(N**i),neps,0.0d0,RHS(i*neq+1:(i+1)*neq,:),neq)
       call dgemm('n','n',neq,neps,neps,-1.0d0,F,neq,(N**(i+1)),neps,-1.0d0,RHS(i*neq+1:(i+1)*neq,:),neq)

    end do


    !call inf_matrices(Ainf,Binf,Cinf,Finf,Ginf,para,neq,neps,npara)


    call qz_solve(Ainf,Binf,Cinf,Finf,Ginf,N,RC,ALPHA,BETA,neq,neps)
    call dgemm('n','n',neq,neps,neps,1.0d0,BETA,neq,(N**max_it),neps,0.0d0,RHS(1+max_it*neq:(1+max_it)*neq,:),neq)

    EYE = 0.0d0
    do gap = 1, neq
       XXX(g0+neq*(gap-1)+1:neq*gap+g0) = -ALPHA(:,gap)
       III(g0+neq*(gap-1)+1:neq*gap+g0) = row_vec + i*neq
       JJJ(g0+neq*(gap-1)+1:neq*gap+g0) = (max_it-1)*neq + gap

       EYE(gap,gap) = 1.0d0
    end do
    g0 = g0 + neq**2


    do gap = 1, neq
       XXX(g0+neq*(gap-1)+1:neq*gap+g0) = EYE(:,gap)
       III(g0+neq*(gap-1)+1:neq*gap+g0) = row_vec + i*neq
       JJJ(g0+neq*(gap-1)+1:neq*gap+g0) = (max_it)*neq + gap
    end do
    g0 = g0 + neq**2
    !print*,g0


    nnz = size(III)

    LHS = 0.0d0
    job = 0
    job(1) = 2
    job(2) = 1
    job(5) = nnz
    job(3) = 1
    job(6) = 0

    call mkl_dcsrcoo(job, (max_it+1)*neq, LHS, ja, ia, nnz, XXX, III, JJJ,info)

    iparm = 0

    error = 0
    msglvl = 0
    mtype = 11
    phase = 13
    mnum = 1
    maxfct = 1


    allocate(RHS2((max_it+1)*neq*neps),MA_VECTOR2((max_it+1)*neq*neps))

    do i = 1,neps
       RHS2((i-1)*(max_it+1)*neq+1:i*(max_it+1)*neq) = RHS(:,i)
    end do

    if (any(isnan(RHS2))) then
       print*,'RHS is NAN'
       print*,RHS
       !print*,ALPHA
       print*,'---------------------'
       do i = 1,npara
          print*, para(i)
       end do
       print*,'--'
       !print*,BETA
       !stop
    end if


    allocate(pt(64))
    do i = 1,64
       pt(i)%DUMMY = 0
    end do

    ! Solve sparse system
    call pardiso(pt, maxfct, mnum, mtype, phase, (max_it+1)*neq, &
         LHS, ia, ja, perm, neps, iparm, msglvl, &
         RHS2, MA_VECTOR2, error)

    do i = 1,neps
       MA_VECTOR(:,i) = MA_VECTOR2((i-1)*(max_it+1)*neq+1:i*(max_it)*neq)
    end do


    ! Release memory 
    phase = -1
    call pardiso (pt, maxfct, mnum, mtype, phase, (max_it+1)*neq, ddum, idum, idum, &
         idum, neps, iparm, msglvl, ddum, ddum, error)


    deallocate(pt,RHS2,MA_VECTOR2)


  end subroutine solve_ma_alt
end module mg
