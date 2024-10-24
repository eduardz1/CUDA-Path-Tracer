cmake_minimum_required(VERSION 3.18...3.28)

project(
    cuda_path_tracer
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_executable(cuda_path_tracer
    apps/main.cpp
)