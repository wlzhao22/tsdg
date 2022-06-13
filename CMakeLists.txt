cmake_minimum_required(VERSION 2.8)
project(TSDG)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS  "-Ofast -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

add_subdirectory(examples)

target_include_directories(scg_build PRIVATE ${CMAKE_SOURCE_DIR})
target_link_directories(scg_build PRIVATE ${CMAKE_BINARY_DIR})

target_include_directories(tsdg_build PRIVATE ${CMAKE_SOURCE_DIR})
target_link_directories(tsdg_build PRIVATE ${CMAKE_BINARY_DIR})

target_include_directories(tsdg_search PRIVATE ${CMAKE_SOURCE_DIR})
target_link_directories(tsdg_search PRIVATE ${CMAKE_BINARY_DIR})

target_include_directories(full_flow PRIVATE ${CMAKE_SOURCE_DIR})
target_link_directories(full_flow PRIVATE ${CMAKE_BINARY_DIR})