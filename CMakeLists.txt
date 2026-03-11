cmake_minimum_required(VERSION 3.15...3.31)
project(shiftclustering LANGUAGES C CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shiftclustering/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shiftclustering/src)

# MeanShiftPP
nanobind_add_module(_meanshiftpp_nb ${SRC_DIR}/_meanshiftpp_nb.cpp)
target_include_directories(_meanshiftpp_nb PRIVATE ${INCLUDE_DIR})
install(TARGETS _meanshiftpp_nb DESTINATION shiftclustering)

# GridShift
nanobind_add_module(_gridshift_nb ${SRC_DIR}/_gridshift_nb.cpp)
target_include_directories(_gridshift_nb PRIVATE ${INCLUDE_DIR})
install(TARGETS _gridshift_nb DESTINATION shiftclustering)

# LocalShift (with OpenMP for parallel per-point processing)
nanobind_add_module(_localshift_nb ${SRC_DIR}/_localshift_nb.cpp)
target_include_directories(_localshift_nb PRIVATE ${INCLUDE_DIR})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_localshift_nb PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS _localshift_nb DESTINATION shiftclustering)

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
