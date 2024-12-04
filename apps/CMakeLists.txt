add_executable(cuda_path_tracer main.cpp)

set_target_properties(cuda_path_tracer
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # CUDA_ARCHITECTURES 90
)

target_compile_features(cuda_path_tracer PRIVATE cxx_std_20)

target_link_libraries(cuda_path_tracer PRIVATE cuda_path_tracer_lib)
