cmake_minimum_required(VERSION 3.12...3.18)
project(jaxpops LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED) # For fftw
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
# E.g. marconi has problems because fftw.h is also in another folder, but env variables are set correctly
if(DEFINED ENV{FFTW_INC})
  set(FFTW_INCLUDEDIR $ENV{FFTW_INC})
endif()
if(DEFINED ENV{FFTW_LIB})
  set(FFTW_LIBDIR $ENV{FFTW_LIB})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${FFTW_INCLUDEDIR} -L${FFTW_LIBDIR} -Wl,-rpath,${FFTW_LIBDIR} -std=c++17 -O3 -lfftw3 -lfftw3f -lfftw3_mpi -lfftw3f_mpi")

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${MPI_INCLUDE_PATH})
include_directories(PkgConfig::FFTW)

# CPU op library
pybind11_add_module(cpu ${CMAKE_CURRENT_LIST_DIR}/lib/cpu.cc)
target_link_libraries(cpu PRIVATE MPI::MPI_CXX)
target_link_libraries(cpu PRIVATE PkgConfig::FFTW)

install(TARGETS cpu DESTINATION jaxpops)

message(STATUS "Building without CUDA")
