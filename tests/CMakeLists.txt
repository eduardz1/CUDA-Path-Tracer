FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)

FetchContent_MakeAvailable(Catch2)

# TODO: Separate them into different test executables
add_executable(testlib
  test_error.cpp
  test_image.cpp
  test_vec3.cpp
  test_ray.cpp
)

set_target_properties(testlib
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 90
)

target_compile_features(testlib PRIVATE cxx_std_20 cuda_std_20)

target_link_libraries(testlib PRIVATE cuda_path_tracer_lib Catch2::Catch2WithMain)

add_test(NAME test_all COMMAND testlib)
