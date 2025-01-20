cmake_minimum_required(VERSION 3.27...3.31)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT DEFINED BUILD_TESTING)
    set(BUILD_TESTING OFF)
endif()

project(
    cuda_path_tracer
    VERSION 1.0
    LANGUAGES CXX CUDA
)

# TODO: add clang-tidy integration in the build once support for CUDA is added
# https://gitlab.kitware.com/cmake/cmake/-/issues/25399

# Include CPM.cmake (CMake Package Manager) to download latest CCCL from NVIDIA
# Github Repository (needed for a couple of features otherwise unsupported in
# Cuda Toolkit 12.0) and Catch2 for testing
include(cmake/CPM.cmake)

CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main
)

# Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/25763
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL})
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if(supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${error}")
endif()

# Only set these properties if this is the main project, not included as a library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    include(CTest)
endif()

include(FetchContent)

add_subdirectory(src)
add_subdirectory(apps)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR CUDA_PATH_TRACER_CMAKE_BUILD_TESTING)
    AND BUILD_TESTING)
    CPMAddPackage("gh:catchorg/Catch2@3.8.0")
    enable_testing()
    add_subdirectory(tests)
endif()
