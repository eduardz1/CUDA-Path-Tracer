set(HEADER_LIST
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/error.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/image.hpp"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/ray.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/render.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shape.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/sphere.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/vec3.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/world.cuh"
)

add_library(cuda_path_tracer_lib
    error.cu
    image.cpp
    ray.cu
    render.cu
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

set_target_properties(cuda_path_tracer_lib
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 90
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
