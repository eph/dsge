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


! to compile
!dlyap :
!	f2py  -c --opt=-O3 -I/opt/intel/mkl/include -I/opt/intel/mkl/include/intel64/lp64/ -L/mq/home/m1eph00/lib/mkl  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_mc -lmkl_p4n -lmkl_mc3 -lmkl_def -lmkl_vml_mc -lmkl_vml_mc3 -liomp5 -lpthread -L/mq/home/m1eph00/lib -lslicot_sequential dlyap_wrapper.f90  --compiler=intel --fcompiler=intelem -m dlyap






















