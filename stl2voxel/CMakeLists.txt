set(CMAKE_CXX_STANDARD 17)
# Set multple architectures here to allow device under different architectures to automatically detect the best one and run the optimal version
# But this will increase the compile time and binary size
set(CMAKE_CUDA_ARCHITECTURES 75;86) 
# If you want to separate the declaration first and then define the function in another file, you need to turn on this option
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(
    voxelizer 
    Voxelizer.cu
    VoxelGrid.cu
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