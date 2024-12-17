add_library(
    voxelizer 
    Voxelizer.cpp
    VoxelGrid.cpp
)

set(EIGEN_DIR /usr/include/eigen3)
target_include_directories(
    voxelizer 
    PUBLIC 
    .
    ${EIGEN_DIR}
)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found")
endif()
if(OpenMP_CXX_FOUND)
    target_compile_options(
        voxelizer
        PUBLIC
        ${OpenMP_CXX_FLAGS}
    )
    target_link_libraries(
        voxelizer
        PUBLIC
        OpenMP::OpenMP_CXX
    )
endif()