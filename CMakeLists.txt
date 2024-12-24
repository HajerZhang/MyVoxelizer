cmake_minimum_required(VERSION 3.20)

project(Voxelizer)

add_subdirectory(tinyxml)
add_subdirectory(stl2voxel)
add_executable(main main.cpp)

target_link_libraries(main PUBLIC voxelizer)

set_target_properties(
    main
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)