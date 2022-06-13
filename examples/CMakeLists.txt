cmake_minimum_required(VERSION 3.5)

project (example)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
# SET(CMAKE_CXX_FLAGS  "-fopenmp")

add_executable(scg_build sc_graph_build.cpp)
add_executable(tsdg_build tsdg_build.cpp)
add_executable(tsdg_search tsdg_search.cpp)
add_executable(full_flow full_flow.cpp)