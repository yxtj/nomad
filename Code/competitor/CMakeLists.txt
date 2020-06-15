cmake_minimum_required(VERSION 2.6)

project(competitor CXX)

ENABLE_LANGUAGE(C)

# find mpi
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})

# find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -Wall -fsigned-char -D TBB_USE_THREADING_TOOLS")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wall -fsigned-char")

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

add_library(util util.cpp)
add_library(smat smat.cpp)
add_executable(mpi-sgd-omp sgd-mpi-omp.cpp)
add_executable(mpi-ccd-r1-omp ccdr1-mpi-omp.cpp)
add_executable(mpi-dsgdpp dsgdpp.cpp)

target_link_libraries(mpi-sgd-omp util)
target_link_libraries(mpi-ccd-r1-omp util)
target_link_libraries(mpi-ccd-r1-omp smat)
target_link_libraries(mpi-dsgdpp util)

