cmake_minimum_required( VERSION 3.0.0 )
project( POINTCLOUD_SH CXX Fortran)

find_package(VTK REQUIRED)
include( ${VTK_USE_FILE} )

find_package(LAPACK REQUIRED)

find_library( SHTOOLS_LIBRARY "SHTOOLS" HINTS ${SHTOOLS_ROOT} )
if(${SHTOOLS_LIBRARY} STREQUAL "SHTOOLS-NOTFOUND")
        message(" SHTOOLS_LIBRARY not found in default locations. Set SHTOOLS_ROOT to specify custom location." )
endif( ${SHTOOLS_LIBRARY} STREQUAL "SHTOOLS-NOTFOUND" )

find_path( SHTOOLS_INCLUDE_DIR "shtools.mod" PATHS ${SHTOOLS_ROOT} PATH_SUFFIXES "include")
if(${SHTOOLS_INCLUDE_DIR} STREQUAL "SHTOOLS_INCLUDE_DIR-NOTFOUND")
        message(" File shtools.mod was not found in default locations. Set SHTOOLS_ROOT to specify custom location." )
endif( ${SHTOOLS_INCLUDE_DIR} STREQUAL "SHTOOLS_INCLUDE_DIR-NOTFOUND" )

add_library( shtoolswrapper STATIC Wrapper.f95 )
target_include_directories( shtoolswrapper PRIVATE ${SHTOOLS_INCLUDE_DIR} )
target_link_libraries( shtoolswrapper PRIVATE ${SHTOOLS_LIBRARY} ${LAPACK_LIBRARIES} )

add_executable( pointCloudSH PointCloudSH.cxx )
target_include_directories( pointCloudSH PRIVATE ${VTK_INCLUDE_DIRS} )
target_link_libraries( pointCloudSH PRIVATE shtoolswrapper ${VTK_LIBRARIES} )

add_dependencies( pointCloudSH shtoolswrapper )
