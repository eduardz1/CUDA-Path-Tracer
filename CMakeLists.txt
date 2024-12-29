cmake_minimum_required(VERSION 3.27...3.31)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

if(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING OFF)
endif()

project(
    cuda_path_tracer
    VERSION 1.0
    LANGUAGES CXX CUDA
)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if (supported)
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
    add_subdirectory(tests)
endif()
