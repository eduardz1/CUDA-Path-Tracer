# TODO: Separate them into different test executables
add_executable(tests
    test_image.cu
    test_ray.cu
    test_rotation.cu
    test_sphere.cu
    test_vec3.cu
)

set_target_properties(tests
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(tests PRIVATE cxx_std_20)
target_link_libraries(tests
    PRIVATE
    Catch2::Catch2WithMain
    CCCL::CCCL
    cuda_path_tracer_lib
)

add_test(tests tests)
