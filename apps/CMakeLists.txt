CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")

add_executable(cuda_path_tracer main.cu)

set_target_properties(cuda_path_tracer
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(cuda_path_tracer PRIVATE cxx_std_20)

target_link_libraries(cuda_path_tracer
    PRIVATE
    cuda_path_tracer_lib
    CLI11::CLI11
)
