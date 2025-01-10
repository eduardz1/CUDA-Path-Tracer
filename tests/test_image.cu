#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <vector_functions.h>

__global__ void wrapperConvertColorTo8Bit(Vec3 *output, const Vec3 color) {
  const auto res = convertColorTo8Bit(color);
  output->x = res.x;
  output->y = res.y;
  output->z = res.z;
}

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Test case for saveImageAsPPM
TEST_CASE("saveImageAsPPM function", "[saveImageAsPPM]") {
  const auto *filename = "test_image.ppm";
  int width = 2;
  int height = 2;
  std::vector<uchar4> image = {
      {255, 0, 0, 255},  // Red
      {0, 255, 0, 255},  // Green
      {0, 0, 255, 255},  // Blue
      {255, 255, 0, 255} // Yellow
  };

  // Call the mock function
  REQUIRE_NOTHROW(saveImageAsPPM(filename, width, height, image));

  // Verify the file content
  std::ifstream file(filename);
  REQUIRE(file.is_open());

  std::string line;
  std::getline(file, line);
  REQUIRE(line == "P3");

  std::getline(file, line);
  REQUIRE(line == "2 2");

  std::getline(file, line);
  REQUIRE(line == "255");

  std::getline(file, line);
  REQUIRE(line == "255 0 0");
  std::getline(file, line);
  REQUIRE(line == "0 255 0");
  std::getline(file, line);
  REQUIRE(line == "0 0 255");
  std::getline(file, line);
  REQUIRE(line == "255 255 0");

  file.close();

  // Clean up
  std::remove(filename);
}

TEST_CASE("convertColorTo8Bit function", "[convertColorTo8Bit]") {
  Vec3 *color = nullptr;
  CUDA_ERROR_CHECK(cudaMallocManaged(&color, sizeof(Vec3)));

  // Test case 1: Normal values
  wrapperConvertColorTo8Bit<<<1, 1>>>(color,
                                      make_float4(0.5F, 0.5F, 0.5F, 1.0F));
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  REQUIRE(color->x == 127);
  REQUIRE(color->y == 127);
  REQUIRE(color->z == 127);

  // Test case 2: Clamping values
  wrapperConvertColorTo8Bit<<<1, 1>>>(color,
                                      make_float4(1.5F, -0.5F, 0.0F, 1.0F));
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  REQUIRE(color->x == 255);
  REQUIRE(color->y == 0);
  REQUIRE(color->z == 0);

  // Test case 3: Edge values
  wrapperConvertColorTo8Bit<<<1, 1>>>(color,
                                      make_float4(1.0F, 1.0F, 1.0F, 1.0F));
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  REQUIRE(color->x == 255);
  REQUIRE(color->y == 255);
  REQUIRE(color->z == 255);

  wrapperConvertColorTo8Bit<<<1, 1>>>(color,
                                      make_float4(0.0F, 0.0F, 0.0F, 1.0F));
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  REQUIRE(color->x == 0);
  REQUIRE(color->y == 0);
  REQUIRE(color->z == 0);

  CUDA_ERROR_CHECK(cudaFree(color));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)