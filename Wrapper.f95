!----------------------------------------------------------------------!
subroutine SHExpandLSQ_Wrapper(cilm, d, lat, lon, nmax, lmax, chi2)
    ! This wrapper ensures that the input arrays have
    ! correct dimensions when calling SHExpandLSQ
    use SHTOOLS, only: SHExpandLSQ

    implicit none
    real*8, intent(in) :: d(nmax), lat(nmax), lon(nmax)
    real*8, intent(out):: cilm(2, lmax + 1, lmax + 1)
    integer,intent(in) :: nmax, lmax
    real*8, intent(out):: chi2
    integer :: norm, csphase, exitstatus
    norm = 4
    csphase = 1
    exitstatus = 0

    call SHExpandLSQ(cilm, d, lat, lon, nmax, lmax, norm, chi2, csphase, &
        exitstatus)

end subroutine SHExpandLSQ_Wrapper

!---------------------------------------------------------------!
subroutine MakeGridPoints_Wrapper(cilm, lmax, n, lat, lon, points, dealloc)
    ! This wrapper ensures that the input arrays have
    ! correct dimensions when calling MakeGridPoint
    use SHTOOLS, only: MakeGridPoint

    implicit none
    integer,intent(in) :: lmax, n, dealloc
    real*8, intent(in) :: cilm(2, lmax + 1, lmax + 1)
    real*8, intent(in) :: lat(n), lon(n)
    real*8, intent(out) :: points(n)
    integer :: norm, csphase, i
    norm = 4
    csphase = 1
    points = 0.0d0

    do i = 1, n
    points(i) = MakeGridPoint( cilm, lmax, lat(i), lon(i),&
        norm, csphase, dealloc)
    end do

end subroutine MakeGridPoints_Wrapper

!---------------------------------------------------------------------!
subroutine GLQGridCoord_Wrapper(latglq, longlq, lmax, nlat, nlong)
    use SHTOOLS, only: GLQGridCoord

    implicit none

    integer, intent(in) :: lmax
    integer, intent(out) :: nlat, nlong
    real*8, intent(out) :: latglq(lmax + 1), longlq(2*lmax + 1)

    call GLQGridCoord(latglq, longlq, lmax, nlat, nlong)

end subroutine GLQGridCoord_Wrapper

!----------------------------------------------------------------------!
subroutine SHGLQ_Wrapper(lmax, zero, w, plx)
    use SHTOOLS, only: SHGLQ

    implicit none

    integer, intent(in) :: lmax
    real*8, intent(out) :: zero(lmax + 1), w(lmax + 1)
    real*8, intent(out) :: plx(1:lmax+1, 1:(lmax+1)*(lmax+2)/2)
    integer :: norm, csphase, cnorm
    norm = 4
    csphase = 1
    cnorm = 0
    call SHGLQ(lmax, zero, w, plx, norm, csphase, cnorm)

end subroutine SHGLQ_Wrapper

!----------------------------------------------------------------------!
subroutine SHExpandGLQ_Wrapper(cilm, lmax, gridglq, w, plx)
    use SHTOOLS, only: SHExpandGLQ

    implicit none

    integer, intent(in) :: lmax
    real*8, intent(in) :: w(lmax + 1), gridglq(lmax + 1, 2*lmax + 1)
    real*8, intent(in) :: plx(1:lmax+1, 1:(lmax+1)*(lmax+2)/2)
    real*8, intent(out) :: cilm(2,lmax + 1,lmax + 1)
    integer :: norm, csphase
    norm = 4
    csphase = 1

    call SHExpandGLQ(cilm, lmax, gridglq, w, plx, norm = norm, &
        csphase = csphase)
end subroutine SHExpandGLQ_Wrapper

!----------------------------------------------------------------------!
subroutine MakeGridGLQ_Wrapper(gridglq, cilm, lmax, plx)
    use SHTOOLS, only: MakeGridGLQ

    implicit none

    integer, intent(in) :: lmax
    real*8, intent(in) :: cilm(2,lmax + 1,lmax + 1)
    real*8, intent(in) :: plx(1:lmax+1, 1:(lmax+1)*(lmax+2)/2)
    real*8, intent(out) :: gridglq(lmax + 1, 2*lmax + 1)
    integer :: norm, csphase
    norm = 4
    csphase = 1

    call MakeGridGLQ(gridglq, cilm, lmax, plx, norm = norm, csphase = csphase)

end subroutine MakeGridGLQ_Wrapper

!----------------------------------------------------------------------!
subroutine SHExpandDH_Wrapper(grid, n, cilm, lmax)
    use SHTOOLS, only: SHExpandDH
    implicit none

    integer, intent(in) :: n
    real*8, intent(in) :: grid(n,n)
    real*8, intent(out) :: cilm(2, n/2, n/2)
    integer, intent(out) :: lmax
    integer :: norm, sampling, csphase, lmax_calc
    norm = 4
    sampling = 1
    csphase = 1

    call SHExpandDH(grid, n, cilm, lmax, norm, sampling, csphase)
end subroutine SHExpandDH_Wrapper

!----------------------------------------------------------------------!
subroutine MakeGridDH_Wrapper(grid, n, cilm, lmax)
    use SHTOOLS, only: MakeGridDH
    implicit none

    integer, intent(in) :: lmax
    real*8, intent(in) :: cilm(2,lmax + 1,lmax + 1)
    integer, intent(out) :: n
    real*8, intent(out) :: grid(2*lmax + 2,2*lmax + 2)
    integer :: norm, sampling, csphase
    norm = 4
    sampling = 1
    csphase = 1

    call MakeGridDH(grid, n, cilm, lmax, norm, sampling, csphase)

end subroutine MakeGridDH_Wrapper

!----------------------------------------------------------------------!
subroutine SHPowerSpectrum_Wrapper(cilm, lmax, pspectrum)
    use SHTOOLS, only: SHPowerSpectrum

    implicit none
    real*8, intent(in) :: cilm(2, lmax + 1, lmax + 1)
    integer, intent(in) :: lmax
    real*8, intent(out) :: pspectrum(lmax + 1)

    call SHPowerSpectrum(cilm, lmax, pspectrum)

end subroutine SHPowerSpectrum_Wrapper

!----------------------------------------------------------------------!
subroutine PlmON_Wrapper(p, lmax, z)
    use SHTools, only: PlmON

    implicit none
    integer, intent(in) :: lmax
    real*8, intent(out) :: p( (lmax + 1)*(lmax + 2)/2 )
    real*8, intent(in)  :: z
    integer :: csphase, cnorm
    csphase = 1
    cnorm = 0

    call PlmON(p, lmax, z, csphase, cnorm)

end subroutine PlmON_Wrapper
