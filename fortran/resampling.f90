module resampling
  

    implicit none

contains

  subroutine multinomial_resampling(npart, wtsim, randu, paraind)

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

  end subroutine multinomial_resampling

end module resampling
