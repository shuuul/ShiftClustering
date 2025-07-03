cmake_minimum_required(VERSION 3.15...3.27)
project(shiftclustering LANGUAGES C CXX)

# Find Python with NumPy - this is the proper way with scikit-build-core
find_package(Python 3.10 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Find Cython
find_program(CYTHON_EXECUTABLE cython)
if(NOT CYTHON_EXECUTABLE)
    # Fallback to using Python module
    set(CYTHON_EXECUTABLE ${Python_EXECUTABLE} -m cython)
endif()

# Set global compile flags for C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add include directory for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/shiftclustering/include)

# Find Cython extensions and build them
file(GLOB CYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shiftclustering/src/*.pyx")

foreach(CYTHON_SOURCE ${CYTHON_SOURCES})
    get_filename_component(MODULE_NAME ${CYTHON_SOURCE} NAME_WE)
    
    # Generate C++ from Cython
    set(CYTHON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.cpp")
    
    # Add explicit Cython compilation command
    add_custom_command(
        OUTPUT ${CYTHON_OUTPUT}
        COMMAND ${CYTHON_EXECUTABLE} --cplus -3 --no-docstrings -X language_level=3 -X boundscheck=False -X wraparound=False ${CYTHON_SOURCE} -o ${CYTHON_OUTPUT}
        DEPENDS ${CYTHON_SOURCE}
        COMMENT "Cythonizing ${MODULE_NAME}.pyx"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Create Python extension module using the new approach
    python_add_library(${MODULE_NAME} MODULE ${CYTHON_OUTPUT} WITH_SOABI)
    
    # Include directories
    target_include_directories(${MODULE_NAME} PRIVATE
        ${Python_NumPy_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/shiftclustering/include
    )
    
    # Install the extension
    install(TARGETS ${MODULE_NAME} DESTINATION shiftclustering)
endforeach()

# Display information
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Found Cython sources: ${CYTHON_SOURCES}") 