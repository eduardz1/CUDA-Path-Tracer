FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0
)

FetchContent_MakeAvailable(Catch2)

# TODO: Separate them into different test executables
add_executable(tests
  test_error.cpp
  test_image.cu
  test_ray.cpp
  test_sphere.cu
  test_vec3.cu
)

include(Catch)

catch_discover_tests(tests)

set_target_properties(tests
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(tests PRIVATE cxx_std_20)

target_link_libraries(tests PRIVATE cuda_path_tracer_lib Catch2::Catch2WithMain)

add_test(NAME test_all COMMAND tests)
