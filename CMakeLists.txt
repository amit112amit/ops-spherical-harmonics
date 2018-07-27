cmake_minimum_required( VERSION 3.0.0 )
project( POINTCLOUD_SH CXX Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

########## Find all the required packages and libraries ##########
find_package(VTK REQUIRED)
include( ${VTK_USE_FILE} )

find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(LAPACK REQUIRED)

find_library( FFTW3_LIBRARY "fftw3" HINTS ${SHTOOLS_ROOT} )

find_library( SHTOOLS_LIBRARY "SHTOOLS" HINTS ${SHTOOLS_ROOT} )
find_path( SHTOOLS_INCLUDE_DIR "shtools.mod" PATHS ${SHTOOLS_ROOT} PATH_SUFFIXES "include")

find_library( SHTNS_LIBRARY "shtns" HINTS ${SHTNS_ROOT} )
find_path( SHTNS_INCLUDE_DIR "shtns.h" PATHS ${SHTNS_ROOT} PATH_SUFFIXES "include")

########## Create all the targets with linking and including info ##########

add_library( shtoolswrapper STATIC Wrapper.f95)
target_include_directories( shtoolswrapper PRIVATE ${SHTOOLS_INCLUDE_DIR} )
target_link_libraries( shtoolswrapper PRIVATE ${SHTOOLS_LIBRARY}
    ${LAPACK_LIBRARIES} ${FFTW3_LIBRARY})

add_executable( pointCloudSH PointCloudSH.cxx )
target_include_directories( pointCloudSH PRIVATE ${VTK_INCLUDE_DIRS} )
target_link_libraries( pointCloudSH PRIVATE shtoolswrapper ${VTK_LIBRARIES}
    Eigen3::Eigen )

add_executable( testCode TestSHE.cxx )
target_include_directories( testCode PRIVATE ${VTK_INCLUDE_DIRS} )
target_link_libraries( testCode PRIVATE shtoolswrapper ${VTK_LIBRARIES}
    Eigen3::Eigen )

add_executable(testShtns TestSHTNS.cxx)
target_link_libraries(testShtns PRIVATE ${VTK_LIBRARIES} Eigen3::Eigen
    ${CGAL_LIBRARIES} ${SHTNS_LIBRARY} ${FFTW3_LIBRARY} m shtoolswrapper)
target_include_directories(testShtns PRIVATE ${SHTNS_INCLUDE_DIR}
    ${CGAL_INCLUDE_DIRS})

########## Declare all the sequence to build the above targets ##########

add_dependencies( pointCloudSH shtoolswrapper )
add_dependencies( testCode shtoolswrapper )
add_dependencies( testShtns shtoolswrapper )
