set(HEADER_LIST
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/error.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/render.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/image.hpp"
)

add_library(cuda_path_tracer_lib
    error.cu
    render.cu
    image.cpp
    vec3.cu
    ray.cu
    ${HEADER_LIST}
)

target_include_directories(cuda_path_tracer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_features(cuda_path_tracer_lib PUBLIC cxx_std_20 cuda_std_20)

set_target_properties(cuda_path_tracer_lib
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
