set(HEADER_LIST
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/parallelogram.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/rectangular_cuboid.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/rotation.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shapes/sphere.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/materials/dielectric.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/materials/lambertian.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/materials/metal.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/camera.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/color.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/error.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/hit_info.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/image.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/material.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/ray.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/scene.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/shape.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/utilities.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/vec3.cuh"
    "${cuda_path_tracer_SOURCE_DIR}/include/cuda_path_tracer/material.cuh"
)

add_library(cuda_path_tracer_lib
    shapes/parallelogram.cu
    shapes/rectangular_cuboid.cu
    shapes/rotation.cu
    shapes/sphere.cu
    materials/dielectric.cu
    materials/lambertian.cu
    materials/metal.cu
    camera.cu
    error.cu
    hit_info.cu
    image.cu
    ray.cu
    scene.cu
    utilities.cu
    vec3.cu
    ${HEADER_LIST}
)

target_include_directories(cuda_path_tracer_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(cuda_path_tracer_lib PUBLIC cxx_std_20)
target_link_libraries(cuda_path_tracer_lib PRIVATE CCCL::CCCL)

if(CMAKE_CUDA_COMPILER MATCHES "nvcc")
    target_compile_options(cuda_path_tracer_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
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
