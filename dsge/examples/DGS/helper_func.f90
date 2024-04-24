function normpdf(z) result(f)

    real(wp), intent(in) :: z
    real(wp) :: f


    f = exp(-z**2/2.0) / sqrt(2.0d0*3.1415926535d0)

  end function normpdf

 

  function normcdf(z) result(f)

   real(wp), intent(in) :: z
    real(wp) :: f
    real(wp) :: t

    ! Constants for the approximation
    real(wp), parameter :: a1 = 0.254829592
    real(wp), parameter :: a2 = -0.284496736
    real(wp), parameter :: a3 = 1.421413741
    real(wp), parameter :: a4 = -1.453152027
    real(wp), parameter :: a5 = 1.061405429
    real(wp), parameter :: p = 0.3275911

    ! Compute the approximation
    t = 1.0_wp / (1.0_wp + p * abs(z))
    f = 1.0_wp - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * exp(-z*z/2.0_wp)

    ! Adjust for z < 0
    if (z < 0.0_wp) then
        f = 1.0_wp - f
    end if

  end function normcdf

 
function norminv(p) result(f)
    ! Variables
    real(wp), intent(in) :: p
    real(wp) :: f
    real(wp) :: q, r

    ! Coefficients for the approximation
    real(wp), parameter :: a1 = -3.969683028665376e+01
    real(wp), parameter :: a2 = 2.209460984245205e+02
    real(wp), parameter :: a3 = -2.759285104469687e+02
    real(wp), parameter :: a4 = 1.383577518672690e+02
    real(wp), parameter :: a5 = -3.066479806614716e+01
    real(wp), parameter :: a6 = 2.506628277459239e+00

    real(wp), parameter :: b1 = -5.447609879822406e+01
    real(wp), parameter :: b2 = 1.615858368580409e+02
    real(wp), parameter :: b3 = -1.556989798598866e+02
    real(wp), parameter :: b4 = 6.680131188771972e+01
    real(wp), parameter :: b5 = -1.328068155288572e+01

    real(wp), parameter :: c1 = -7.784894002430293e-03
    real(wp), parameter :: c2 = -3.223964580411365e-01
    real(wp), parameter :: c3 = -2.400758277161838e+00
    real(wp), parameter :: c4 = -2.549732539343734e+00
    real(wp), parameter :: c5 = 4.374664141464968e+00
    real(wp), parameter :: c6 = 2.938163982698783e+00

    real(wp), parameter :: d1 = 7.784695709041462e-03
    real(wp), parameter :: d2 = 3.224671290700398e-01
    real(wp), parameter :: d3 = 2.445134137142996e+00
    real(wp), parameter :: d4 = 3.754408661907416e+00

    ! Compute the approximation
    if (p < 0.02425_wp) then
        ! Lower tail
        q = sqrt(-2.0_wp * log(p))
        f = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0_wp)
    else if (p > 0.97575_wp) then
        ! Upper tail
        q = sqrt(-2.0_wp * log(1.0_wp - p))
        f = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0_wp)
    else
        ! Central range
        q = p - 0.5_wp
        r = q * q
        f = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0_wp)
    end if
end function norminv

 

  function zetaspbfcn(z,sigma,sprd) result(f)

    real(wp), intent(in) :: z,sigma,sprd

    real(wp) :: zetaratio, nk

    real(wp) :: f

 

    zetaratio = zetabomegafcn(z,sigma,sprd)/zetazomegafcn(z,sigma,sprd);

    nk = nkfcn(z,sigma,sprd);

    f = -zetaratio/(1-zetaratio)*nk/(1-nk);

  end function zetaspbfcn

 

  function zetabomegafcn(z,sigma,sprd) result(f)

    real(wp), intent(in) :: z,sigma,sprd

    real(wp) :: nk, mustar,omegastar,Gammastar,Gstar,dGammadomegastar

    real(wp) :: dGdomegastar, d2Gammadomega2star,d2Gdomega2star

    real(wp) :: f

 

    nk = nkfcn(z,sigma,sprd);

    mustar = mufcn(z,sigma,sprd);

    omegastar = omegafcn(z,sigma);

    Gammastar = Gammafcn(z,sigma);

    Gstar = Gfcn(z,sigma);

    dGammadomegastar = dGammadomegafcn(z);

    dGdomegastar = dGdomegafcn(z,sigma);

    d2Gammadomega2star = d2Gammadomega2fcn(z,sigma);

    d2Gdomega2star = d2Gdomega2fcn(z,sigma);

    f = omegastar*mustar*nk*(d2Gammadomega2star*dGdomegastar-d2Gdomega2star*dGammadomegastar)/(dGammadomegastar-mustar*dGdomegastar)**2/sprd/(1.0d0-Gammastar+dGammadomegastar*(Gammastar-mustar*Gstar)/(dGammadomegastar-mustar*dGdomegastar));

  end function zetabomegafcn

 

  function zetazomegafcn(z,sigma,sprd) result(f)

    real(wp), intent(in) :: z,sigma,sprd

    real(wp) :: mustar

    real(wp) :: f

 

    mustar = mufcn(z,sigma,sprd);

    f = omegafcn(z,sigma)*(dGammadomegafcn(z)-mustar*dGdomegafcn(z,sigma))/(Gammafcn(z,sigma)-mustar*Gfcn(z,sigma));

  end function zetazomegafcn

 

  function nkfcn(z,sigma,sprd) result(f)

    real(wp), intent(in) :: z,sigma,sprd

    real(wp) :: f

 

    f = 1.0d0-(Gammafcn(z,sigma)-mufcn(z,sigma,sprd)*Gfcn(z,sigma))*sprd;

  end function nkfcn

 

  function mufcn(z,sigma,sprd) result(f)

    real(wp), intent(in) :: z,sigma,sprd

    real(wp) :: f

 

    f = (1.0d0-1.0d0/sprd)/(dGdomegafcn(z,sigma)/dGammadomegafcn(z)*(1.0d0-Gammafcn(z,sigma))+Gfcn(z,sigma));

  end function mufcn

 

  function omegafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = exp(sigma*z-1.0d0/2.0d0*sigma**2);

  end function omegafcn

 

  function Gfcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = normcdf(z-sigma);

  end function Gfcn

 

  function Gammafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = omegafcn(z,sigma)*(1.0d0-normcdf(z))+normcdf(z-sigma);

  end function Gammafcn

 

  function dGdomegafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f=normpdf(z)/sigma;

  end function dGdomegafcn

 

  function d2Gdomega2fcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = -z*normpdf(z)/omegafcn(z,sigma)/sigma**2;

  end function d2Gdomega2fcn

 

  function dGammadomegafcn(z) result(f)

    real(wp), intent(in) :: z

    real(wp) :: f

 

    f = 1.0d0-normcdf(z);

  end function dGammadomegafcn

 

  function d2Gammadomega2fcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = -normpdf(z)/omegafcn(z,sigma)/sigma;

  end function d2Gammadomega2fcn

 

  function dGdsigmafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = -z*normpdf(z-sigma)/sigma;

  end function dGdsigmafcn

 

 

  function d2Gdomegadsigmafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = -normpdf(z)*(1.0d0-z*(z-sigma))/(sigma**2)

 

  end function d2Gdomegadsigmafcn

 

  function dGammadsigmafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = -normpdf(z-sigma)

 

  end function dGammadsigmafcn

 

  function d2Gammadomegadsigmafcn(z,sigma) result(f)

    real(wp), intent(in) :: z,sigma

    real(wp) :: f

 

    f = (z/sigma-1.0d0)*normpdf(z)

 

  end function d2Gammadomegadsigmafcn

 

 

  function get_sigwstar(zwstar, sprd, zeta_spb) result(x)

    real(wp), intent(in) :: zwstar, sprd, zeta_spb

    real(wp) :: x

 

    real(wp) :: f, x1,x2,f1,f2, tol

    integer :: tries, max_tries

 

    x = 0.508949153597541d0

 

    x1 = 0.01d0

    x2 = 1.0d0

 

    f1 = zetaspbfcn(zwstar, x1, sprd)-zeta_spb

    f2 = zetaspbfcn(zwstar, x2, sprd)-zeta_spb

 

    if (f1*f2 > 0.0d0) then

       print*,'BAD FZERO BOUNDS'

       x = -100.0;

    end if

 

 

    max_tries = 500

    tol = 1.0e-13

    ! very simple bisection algorithm

    do tries = 1,max_tries

       x = (x1 + x2) / 2.0d0

 

       f = zetaspbfcn(zwstar, x, sprd)-zeta_spb

 

       if (f*f1>0.0d0) then

          f1 = f; x1 = x

       else

          f2 = f; x2 = x

       end if

 

       if (abs(f2-f1) < tol) then

          return

       end if

    end do

  end function get_sigwstar
