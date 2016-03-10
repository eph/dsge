module gensys_wrapper


  !use mkl95_precision, only: wp => dp
  use gensys


  implicit none


contains

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

    ddiv = div


    call do_gensys(TT, CC, RR, fmat, fwt, ywt, gev, eu, loose, &
         ns, neps, neta, &
       GG0, GG1, CC0, PPSI, PPI,ddiv)


  end subroutine call_gensys


end module gensys_wrapper
