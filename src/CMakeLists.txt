set(HEADER_LIST
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/parallelogram.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/rectangular_cuboid.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/rotation.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/sphere.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/camera.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/error.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/hit_info.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/image.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/ray.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/scene.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shape.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/utilities.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/vec3.cuh"
)

add_library(cuda_path_tracer_lib
    shapes/parallelogram.cu
    shapes/rectangular_cuboid.cu
    shapes/rotation.cu
    shapes/sphere.cu
    camera.cu
    error.cu
    hit_info.cu
    image.cu
    ray.cu
    scene.cu
    vec3.cu
    ${HEADER_LIST}
)

target_include_directories(cuda_path_tracer_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} # Explicit for clang
)

target_compile_features(cuda_path_tracer_lib PUBLIC cxx_std_20)

if(CMAKE_CUDA_COMPILER MATCHES "nvcc")
    target_compile_options(cuda_path_tracer_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # FIXME: Ugly workaround for issue https://github.com/llvm/llvm-project/issues/95183 remove once fixed
    target_compile_options(cuda_path_tracer_lib PRIVATE --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13)
endif()

set_target_properties(cuda_path_tracer_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)
