#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/catch_test_macros.hpp>

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// Test case for ray constructors
TEST_CASE("ray constructors", "[ray]") {
  Ray r1;
  REQUIRE(r1.getOrigin() == Vec3(0, 0, 0));
  REQUIRE(r1.getDirection() == Vec3(0, 0, 0));

  Vec3 origin(1.0F, 2.0F, 3.0F);
  Vec3 direction(4.0F, 5.0F, 6.0F);
  Ray r2(origin, direction);
  REQUIRE(r2.getOrigin() == origin);
  REQUIRE(r2.getDirection() == direction);
}

// Test case for ray::getOrigin
TEST_CASE("ray::getOrigin", "[ray]") {
  Vec3 origin(1.0F, 2.0F, 3.0F);
  Vec3 direction(4.0F, 5.0F, 6.0F);
  Ray r(origin, direction);
  REQUIRE(r.getOrigin() == origin);
}

// Test case for ray::getDirection
TEST_CASE("ray::getDirection", "[ray]") {
  Vec3 origin(1.0F, 2.0F, 3.0F);
  Vec3 direction(4.0F, 5.0F, 6.0F);
  Ray r(origin, direction);
  REQUIRE(r.getDirection() == direction);
}

// Test case for ray::at
TEST_CASE("ray::at", "[ray]") {
  Vec3 origin(1.0F, 2.0F, 3.0F);
  Vec3 direction(4.0F, 5.0F, 6.0F);
  Ray r(origin, direction);
  float t = 2.0F;
  Vec3 expected_position = origin + direction * t;
  REQUIRE(r.at(t) == expected_position);
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)