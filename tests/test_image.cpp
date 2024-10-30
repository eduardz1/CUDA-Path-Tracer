#include "cuda_path_tracer/image.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <vector_functions.h>

// Test case for saveImageAsPPM
TEST_CASE("saveImageAsPPM function", "[saveImageAsPPM]") {
  auto filename = "test_image.ppm";
  int width = 2;
  int height = 2;
  uchar4 image[4] = {
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
  // Test case 1: Normal values
  uchar4 color = convertColorTo8Bit(make_float4(0.5f, 0.5f, 0.5f, 1.0f));
  REQUIRE(color.x == 127);
  REQUIRE(color.y == 127);
  REQUIRE(color.z == 127);
  REQUIRE(color.w == 255);

  // Test case 2: Clamping values
  color = convertColorTo8Bit(make_float4(1.5f, -0.5f, 0.0f, 1.0f));
  REQUIRE(color.x == 255);
  REQUIRE(color.y == 0);
  REQUIRE(color.z == 0);
  REQUIRE(color.w == 255);

  // Test case 3: Edge values
  color = convertColorTo8Bit(make_float4(1.0f, 1.0f, 1.0f, 1.0f));
  REQUIRE(color.x == 255);
  REQUIRE(color.y == 255);
  REQUIRE(color.z == 255);
  REQUIRE(color.w == 255);

  color = convertColorTo8Bit(make_float4(0.0f, 0.0f, 0.0f, 1.0f));
  REQUIRE(color.x == 0);
  REQUIRE(color.y == 0);
  REQUIRE(color.z == 0);
  REQUIRE(color.w == 255);
}