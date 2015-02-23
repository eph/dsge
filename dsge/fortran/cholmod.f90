module cholmod
  implicit none
  integer, parameter :: wp = selected_real_kind(15)
  !use mkl95_precision, only: wp => dp


contains
  subroutine do_cholmod(AA, L, indef,nn)
    ! Performs of a modified Cholesky decomposition on A such that L*L'- A is minimized.
    !--------------------------------------------------------------------------------
    integer, intent(in) ::  nn
    real(wp), intent(in) :: AA(nn,nn)
    real(wp), intent(out) :: L(nn, nn)
    integer, intent(out) :: indef

    real(wp), external :: ddot

    integer :: n, i, j
    real(wp) :: diagA(nn), R(nn, nn), d(nn)
    real(wp) :: gamma, xi, beta, delta, theta, djtemp, A(nn,nn)

    real(wp), allocatable :: ccol(:)

    real(wp) :: REALLY_SMALL


    A = AA
    ! symmetrize
    A = 0.5_wp*(A + transpose(A))
    L = A
    call dpotrf('l',size(A,1), L, size(A,1), indef)

      if (indef == 0) then
         !return
      end if
      L = 0.0_wp
      REALLY_SMALL = epsilon(REALLY_SMALL)

      REALLY_SMALL = 2.2204e-016
      n = size(A, 1)

      R = A
      do i = 1,n
         diagA = A(i,i)
         R(i,i) = 0.0_wp
         L(i,i)=1.0_wp
      end do

      gamma = maxval(abs(diagA))
      xi = maxval(abs(R))
      delta = REALLY_SMALL*maxval((/ gamma+xi, 1.0_wp/))
      beta = sqrt(maxval((/gamma, xi/(n*1.0_wp), REALLY_SMALL/)))

      indef = 0

      do j = 1, n

         if (j == 1) then
            djtemp = A(j,j)

            if (j < n) then
               allocate(ccol(n-j))
               ccol = A(j+1:n,j)
               theta = maxval(abs(ccol))
               d(j) = maxval((/abs(djtemp), (theta/beta)**2, delta/))
               L(j+1:n, j) = ccol / d(j)
               deallocate(ccol)
            else
               d(j) = maxval((/abs(djtemp), delta/))
            end if
         else

            djtemp = A(j,j) - ddot(j-1, L(j,1:j-1), 1, d(1:j-1) * L(j,1:j-1), 1)

            if (j < n) then
               allocate(ccol(n-j))
               !            print*, 'A = ', A(j+1:n, j)
               ccol = A(j+1:n, j) - matmul(L(j+1:n,1:j-1), d(1:j-1) * L(j,1:j-1))
               !           print*,ccol
               theta = maxval(abs(ccol))
               !            print*,'theta', theta
               d(j) = maxval((/abs(djtemp), (theta/beta)**2, delta/))
               L(j+1:n, j) = ccol / d(j)

               deallocate(ccol)
            else

               d(j) = maxval((/abs(djtemp), delta/))

            end if

         end if

         if (d(j) > djtemp) then

            indef = 1

         end if

      end do

      do j = 1, n

         L(:,j) = L(:,j) * sqrt(d(j))

      end do


    end subroutine do_cholmod
end module cholmod
