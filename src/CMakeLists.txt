set(HEADER_LIST
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/camera.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/error.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/image.hpp"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/ray.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/scene.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shape.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/sphere.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/vec3.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/world.cuh"
)

add_library(cuda_path_tracer_lib

    # SHARED
    camera.cu
    error.cu
    image.cpp
    ray.cu
    scene.cu
    sphere.cu
    vec3.cu
    world.cu
    ${HEADER_LIST}
)

target_include_directories(cuda_path_tracer_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} # Explicit for clang
)

target_compile_features(cuda_path_tracer_lib PUBLIC cxx_std_20)

if(CMAKE_CUDA_COMPILER MATCHES "nvcc")
    target_compile_options(cuda_path_tracer_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

set_target_properties(cuda_path_tracer_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON

    # CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES all
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
