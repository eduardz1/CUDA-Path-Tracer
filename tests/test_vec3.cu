#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

const auto epsilon = 1e-6f;

__global__ void testFloat4Conversion(Vec3 input, float4 *output) {
  float4 result = input;
  *output = result;
}

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers)

// Test case for vec3 constructors
TEST_CASE("vec3 constructors", "[vec3]") {
  Vec3 v1;
  REQUIRE(v1.x == 0.0f);
  REQUIRE(v1.y == 0.0f);
  REQUIRE(v1.z == 0.0f);

  Vec3 v2(1.0f);
  REQUIRE(v2.x == 1.0f);
  REQUIRE(v2.y == 1.0f);
  REQUIRE(v2.z == 1.0f);

  Vec3 v3(1.0f, 2.0f, 3.0f);
  REQUIRE(v3.x == 1.0f);
  REQUIRE(v3.y == 2.0f);
  REQUIRE(v3.z == 3.0f);
}

// Test case for vec3 addition
TEST_CASE("vec3 addition", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  Vec3 v3 = v1 + v2;
  REQUIRE(v3.x == 5.0f);
  REQUIRE(v3.y == 7.0f);
  REQUIRE(v3.z == 9.0f);
}

// Test case for vec3 subtraction
TEST_CASE("vec3 subtraction", "[vec3]") {
  Vec3 v1(4.0f, 5.0f, 6.0f);
  Vec3 v2(1.0f, 2.0f, 3.0f);
  Vec3 v3 = v1 - v2;
  REQUIRE(v3.x == 3.0f);
  REQUIRE(v3.y == 3.0f);
  REQUIRE(v3.z == 3.0f);
}

// Test case for vec3 multiplication by vector
TEST_CASE("vec3 multiplication", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  Vec3 v3 = v1 * v2;
  REQUIRE(v3.x == 4.0f);
  REQUIRE(v3.y == 10.0f);
  REQUIRE(v3.z == 18.0f);
}

// Test case for vec3 multiplication by scalar
TEST_CASE("vec3 multiplication by scalar", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  float scalar = 2.0f;
  Vec3 v2 = v1 * scalar;
  REQUIRE(v2.x == 2.0f);
  REQUIRE(v2.y == 4.0f);
  REQUIRE(v2.z == 6.0f);
}

// Test case for vec3 equality
TEST_CASE("vec3 equality", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(1.0f, 2.0f, 3.0f);
  REQUIRE(v1 == v2);
}

// Test case for vec3 dot product
TEST_CASE("vec3 dot product", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  float dotProduct = dot(v1, v2);
  REQUIRE(dotProduct == 32.0f);
}

// Test case for vec3 >> operator
TEST_CASE("vec3 >> operator", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  std::stringstream ss;
  ss << v1;
  REQUIRE(ss.str() == "(1, 2, 3)");
}

// Test case for vec3 cross product
TEST_CASE("vec3 cross product", "[vec3]") {
  Vec3 v1(2.0f, 3.0f, 4.0f);
  Vec3 v2(5.0f, 6.0f, 7.0f);
  Vec3 v3 = cross(v1, v2);
  REQUIRE(v3.x == -3.0f);
  REQUIRE(v3.y == 6.0f);
  REQUIRE(v3.z == -3.0f);
}

// Test case for vec3 negation
TEST_CASE("vec3 negation", "[vec3]") {
  Vec3 v1(1.0f, -2.0f, 3.0f);
  Vec3 v2 = -v1;
  REQUIRE(v2.x == -1.0f);
  REQUIRE(v2.y == 2.0f);
  REQUIRE(v2.z == -3.0f);
}

// Test case for vec3 division by scalar
TEST_CASE("vec3 division by scalar", "[vec3]") {
  Vec3 v1(2.0f, 4.0f, 6.0f);
  float scalar = 2.0f;
  Vec3 v2 = v1 / scalar;
  REQUIRE(v2.x == 1.0f);
  REQUIRE(v2.y == 2.0f);
  REQUIRE(v2.z == 3.0f);
}

// Test case for vec3 length calculations
TEST_CASE("vec3 length calculations", "[vec3]") {
  Vec3 v1(3.0f, 4.0f, 0.0f);
  REQUIRE(v1.getLengthSquared() == 25.0f);
  REQUIRE(v1.getLength() == 5.0f);
}

// Test case for vec3 unit vector
TEST_CASE("vec3 unit vector", "[vec3]") {
  Vec3 v1(3.0f, 4.0f, 0.0f);
  Vec3 unit = makeUnitVector(v1);
  REQUIRE_THAT(unit.x, Catch::Matchers::WithinAbs(0.6f, epsilon));
  REQUIRE_THAT(unit.y, Catch::Matchers::WithinAbs(0.8f, epsilon));
  REQUIRE_THAT(unit.z, Catch::Matchers::WithinAbs(0.0f, epsilon));
}

// Test case for vec3 += operator
TEST_CASE("vec3 += operator", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  v1 += v2;
  REQUIRE(v1.x == 5.0f);
  REQUIRE(v1.y == 7.0f);
  REQUIRE(v1.z == 9.0f);
}

// Test case for vec3 float4 conversion
TEST_CASE("vec3 float4 conversion", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  float4 *f4 = nullptr;
  CUDA_ERROR_CHECK(cudaMallocManaged(&f4, sizeof(float4)));
  testFloat4Conversion<<<1, 1>>>(v1, f4);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaGetLastError());

  REQUIRE(f4->x == 1.0f);
  REQUIRE(f4->y == 2.0f);
  REQUIRE(f4->z == 3.0f);
  REQUIRE(f4->w == 1.0f);

  CUDA_ERROR_CHECK(cudaFree(f4));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers)