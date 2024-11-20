cmake_minimum_required(VERSION 3.1.0)
project(robust-laplacian-py)

# Recurse in to pybind
set(PYBIND11_NEWPYTHON ON)
add_subdirectory(deps/pybind11)

# set location of eigen for geometry-central
set(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen" CACHE PATH "my path")

# geometry-central
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/geometry-central)

pybind11_add_module(robust_laplacian_bindings 
  src/cpp/point_cloud_utilities.cpp
  src/cpp/core.cpp
)

include_directories(robust_laplacian_bindings ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
include_directories(robust_laplacian_bindings ${CMAKE_CURRENT_SOURCE_DIR}/deps/jc_voronoi/include)

target_link_libraries(robust_laplacian_bindings PRIVATE geometry-central)

install(TARGETS robust_laplacian_bindings LIBRARY DESTINATION .)