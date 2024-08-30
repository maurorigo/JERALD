cmake_minimum_required(VERSION 3.12...3.18)
project(jaxpops LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED) # For fftw
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{FFTW_INC} -L$ENV{FFTW_LIB} -std=c++17 -O3 -lfftw3 -lfftw3f -lfftw3_mpi -lfftw3f_mpi")

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${MPI_INCLUDE_PATH})
include_directories(PkgConfig::FFTW)

# CPU op library
pybind11_add_module(cpu ${CMAKE_CURRENT_LIST_DIR}/lib/cpu.cc)
target_link_libraries(cpu PRIVATE MPI::MPI_CXX)
target_link_libraries(cpu PRIVATE PkgConfig::FFTW)
target_link_libraries(cpu PRIVATE fftw3)

install(TARGETS cpu DESTINATION jaxpops)

message(STATUS "Building without CUDA")
