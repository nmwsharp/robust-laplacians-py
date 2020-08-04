cmake_minimum_required(VERSION 3.1.0)
project(robust-laplacian-py)

# Recurse in to pybind
add_subdirectory(deps/pybind11)

# set location of eigen for geometry-central
set(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen" CACHE PATH "my path")

# geometry-central
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/geometry-central)

pybind11_add_module(robust_laplacian_bindings 
  src/cpp/core.cpp
)

target_link_libraries(robust_laplacian_bindings PRIVATE geometry-central)
