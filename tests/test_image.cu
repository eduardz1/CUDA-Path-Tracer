#include "cuda_path_tracer/image.cuh"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <vector_functions.h>

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Test case for saveImageAsPPM
TEST_CASE("saveImageAsPPM function", "[saveImageAsPPM]") {
  const auto *filename = "test_image.ppm";
  int width = 2;
  int height = 2;
  std::vector<uchar4> image = {
      {.x = 255, .y = 0, .z = 0, .w = 255},  // Red
      {.x = 0, .y = 255, .z = 0, .w = 255},  // Green
      {.x = 0, .y = 0, .z = 255, .w = 255},  // Blue
      {.x = 255, .y = 255, .z = 0, .w = 255} // Yellow
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

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)