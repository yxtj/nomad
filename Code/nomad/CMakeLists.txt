cmake_minimum_required(VERSION 2.6)

project(nomad CXX)

# Project version
set(NOMAD_MAJOR 1)
set(NOMAD_MINOR 0)

set(CMAKE_CXX_STANDARD 11)

# find mpi
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
# BUGBUG: is this quotation mark really needed? if I don't put it, it generates semicolons
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})
message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")

# Locate TBB

# Add FindTBB directory to CMake's module path
# This cmake module is from http://code.google.com/p/findtbb
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTBB/")

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
link_libraries(${TBB_LIBRARIES})
message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
	message(STATUS "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTBB_USE_DEBUG=0 -DTBB_IMPLEMENT_CPP0X=1")
	#add_definitions(-DTBB_USE_THREADING_TOOLS)
	#add_definitions(-DTBB_USE_DEBUG=0)
else ()
	message(STATUS "Complier: ${CMAKE_CXX_COMPILER_ID}")
	#set(CMAKE_CXX_FLAGS_DEBUG "-Wall -fsigned-char -D TBB_USE_THREADING_TOOLS")
	set(CMAKE_CXX_FLAGS_DEBUG "-Wall -fsigned-char -DTBB_USE_DEBUG=0 -DTBB_IMPLEMENT_CPP0X=1")
	set(CMAKE_CXX_FLAGS_RELEASE "-Wall -fsigned-char")
endif ()


# Locate Boost
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)
include_directories( ${Boost_INCLUDE_DIR} )
link_libraries(${Boost_LIBRARIES} )
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

# Locate Thread library
find_package (Threads)
message(STATUS "Thread lib: ${CMAKE_THREAD_LIBS_INIT}")

add_executable(nomad
	main.cpp
	nomad_option.cpp pool.cpp
	nb_data.cpp nb_init.cpp nb_main.cpp nb_master.cpp nb_sendrecv.cpp
	nb_net_control.cpp  nb_checkpoint.cpp
)
target_link_libraries(nomad ${CMAKE_THREAD_LIBS_INIT})

add_executable(synthgen synthgen.cpp)
add_executable(synthgen_conf synthgen-conf.cpp)
#add_executable(permute permute.cpp)
