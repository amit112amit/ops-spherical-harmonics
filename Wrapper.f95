subroutine SHExpandLSQ_Wrapper(cilm, d, lat, lon, nmax, &
                lmax, norm, chi2, csphase, exitstatus)
        ! This wrapper ensures that the input arrays have
        ! correct dimensions when calling SHExpandLSQ
        use SHTOOLS, only: SHExpandLSQ

        implicit none
        real*8, intent(in) :: d(nmax), lat(nmax), lon(nmax)
        real*8, intent(out):: cilm(2, lmax + 1, lmax + 1)
        integer,intent(in) :: nmax, lmax
        integer,intent(in) :: norm, csphase
        real*8, intent(out):: chi2
        integer,intent(out):: exitstatus
        integer :: l, m, i

        call SHExpandLSQ( cilm, d, lat, lon, nmax, lmax, &
                norm, chi2, csphase, exitstatus)

end subroutine SHExpandLSQ_Wrapper

!---------------------------------------------------------------!
function MakeGridPoint_Wrapper(cilm, lmax, lat, lon, norm,&
                csphase, dealloc) result(value)
        ! This wrapper ensures that the input arrays have
        ! correct dimensions when calling MakeGridPoint
        use SHTOOLS, only: MakeGridPoint

        implicit none
        integer,intent(in) :: lmax
        integer,intent(in) :: norm, csphase, dealloc
        real*8, intent(in) :: cilm(2, lmax + 1, lmax + 1)
        real*8, intent(in) :: lat, lon
        real*8 :: value

        value = MakeGridPoint( cilm, lmax, lat, lon, norm, &
                        csphase, dealloc)

end function MakeGridPoint_Wrapper
