#include "cuda_path_tracer/vec3.cuh"
#include <catch2/catch_test_macros.hpp>

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers)

// Test case for vec3 constructors
TEST_CASE("vec3 constructors", "[vec3]") {
  Vec3 v1;
  REQUIRE(v1.getX() == 0.0f);
  REQUIRE(v1.getY() == 0.0f);
  REQUIRE(v1.getZ() == 0.0f);

  Vec3 v2(1.0f);
  REQUIRE(v2.getX() == 1.0f);
  REQUIRE(v2.getY() == 1.0f);
  REQUIRE(v2.getZ() == 1.0f);

  Vec3 v3(1.0f, 2.0f, 3.0f);
  REQUIRE(v3.getX() == 1.0f);
  REQUIRE(v3.getY() == 2.0f);
  REQUIRE(v3.getZ() == 3.0f);
}

// Test case for vec3 addition
TEST_CASE("vec3 addition", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  Vec3 v3 = v1 + v2;
  REQUIRE(v3.getX() == 5.0f);
  REQUIRE(v3.getY() == 7.0f);
  REQUIRE(v3.getZ() == 9.0f);
}

// Test case for vec3 subtraction
TEST_CASE("vec3 subtraction", "[vec3]") {
  Vec3 v1(4.0f, 5.0f, 6.0f);
  Vec3 v2(1.0f, 2.0f, 3.0f);
  Vec3 v3 = v1 - v2;
  REQUIRE(v3.getX() == 3.0f);
  REQUIRE(v3.getY() == 3.0f);
  REQUIRE(v3.getZ() == 3.0f);
}

// Test case for vec3 multiplication by vector
TEST_CASE("vec3 multiplication", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  Vec3 v2(4.0f, 5.0f, 6.0f);
  Vec3 v3 = v1 * v2;
  REQUIRE(v3.getX() == 4.0f);
  REQUIRE(v3.getY() == 10.0f);
  REQUIRE(v3.getZ() == 18.0f);
}

// Test case for vec3 multiplication by scalar
TEST_CASE("vec3 multiplication by scalar", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  float scalar = 2.0f;
  Vec3 v2 = v1 * scalar;
  REQUIRE(v2.getX() == 2.0f);
  REQUIRE(v2.getY() == 4.0f);
  REQUIRE(v2.getZ() == 6.0f);
}

// Test case for vec3 division
TEST_CASE("vec3 division", "[vec3]") {
  Vec3 v1(4.0f, 6.0f, 8.0f);
  Vec3 v2(2.0f, 3.0f, 4.0f);
  Vec3 v3 = v1 / v2;
  REQUIRE(v3.getX() == 2.0f);
  REQUIRE(v3.getY() == 2.0f);
  REQUIRE(v3.getZ() == 2.0f);
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
  float dotProduct = v1.dot(v2);
  REQUIRE(dotProduct == 32.0f);
}

// Test case for vec3 >> operator
TEST_CASE("vec3 >> operator", "[vec3]") {
  Vec3 v1(1.0f, 2.0f, 3.0f);
  std::stringstream ss;
  ss << v1;
  REQUIRE(ss.str() == "(1, 2, 3)");
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers)