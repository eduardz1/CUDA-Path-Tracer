FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)

FetchContent_MakeAvailable(Catch2)

# TODO: Separate them into different test executables
add_executable(tests
  test_error.cpp
  test_image.cpp
  test_vec3.cpp
  test_ray.cpp
)

include(Catch)

catch_discover_tests(tests)

set_target_properties(tests
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTUREs 50
)

target_compile_features(tests PRIVATE cxx_std_20)

target_link_libraries(tests PRIVATE cuda_path_tracer_lib Catch2::Catch2WithMain)

add_test(NAME test_all COMMAND tests)
