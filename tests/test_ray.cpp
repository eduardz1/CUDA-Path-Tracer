#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/catch_test_macros.hpp>

// Test case for ray constructors
TEST_CASE("ray constructors", "[ray]") {
  ray r1;
  REQUIRE(r1.getOrigin() == vec3(0, 0, 0));
  REQUIRE(r1.getDirection() == vec3(0, 0, 0));

  vec3 origin(1.0f, 2.0f, 3.0f);
  vec3 direction(4.0f, 5.0f, 6.0f);
  ray r2(origin, direction);
  REQUIRE(r2.getOrigin() == origin);
  REQUIRE(r2.getDirection() == direction);
}

// Test case for ray::getOrigin
TEST_CASE("ray::getOrigin", "[ray]") {
  vec3 origin(1.0f, 2.0f, 3.0f);
  vec3 direction(4.0f, 5.0f, 6.0f);
  ray r(origin, direction);
  REQUIRE(r.getOrigin() == origin);
}

// Test case for ray::getDirection
TEST_CASE("ray::getDirection", "[ray]") {
  vec3 origin(1.0f, 2.0f, 3.0f);
  vec3 direction(4.0f, 5.0f, 6.0f);
  ray r(origin, direction);
  REQUIRE(r.getDirection() == direction);
}

// Test case for ray::at
TEST_CASE("ray::at", "[ray]") {
  vec3 origin(1.0f, 2.0f, 3.0f);
  vec3 direction(4.0f, 5.0f, 6.0f);
  ray r(origin, direction);
  float t = 2.0f;
  vec3 expected_position = origin + direction * t;
  REQUIRE(r.at(t) == expected_position);
}