add_executable(tests
    test_image.cu
    test_parallelogram.cu
    test_random_in_unit_disk.cu
    test_ray.cu
    test_rectangular_cuboid.cu
    test_render_bench.cu
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
    cuda_path_tracer_lib
)

add_test(tests tests)
