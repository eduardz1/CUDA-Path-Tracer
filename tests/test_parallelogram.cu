#include "catch2/matchers/catch_matchers.hpp"
#include "cuda_path_tracer/error.cuh"
#define protected public
#include "cuda_path_tracer/shapes/parallelogram.cuh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

const auto epsilon = 1e-6F;

namespace {
__global__ void testParallelogramHit(const Parallelogram *parallelogram,
                                     const Ray *ray, float t_min, float t_max,
                                     bool *hit_result, HitInfo *hit_info) {
  *hit_result = parallelogram->hit(*ray, t_min, t_max, *hit_info);
}
} // namespace

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST_CASE("Parallelogram Construction and Basic Properties",
          "[parallelogram]") {
  SECTION("Default construction") {
    Parallelogram p;
    // Default construction should not crash
    REQUIRE_NOTHROW(p);
  }

  SECTION("Construction with parameters") {
    Vec3 origin(0.0F, 0.0F, 0.0F);
    Vec3 u(1.0F, 0.0F, 0.0F);
    Vec3 v(0.0F, 1.0F, 0.0F);

    Parallelogram p(origin, u, v);
    // Construction with parameters should not crash
    REQUIRE_NOTHROW(p);
  }
}

TEST_CASE("Parallelogram Ray Intersection Tests", "[parallelogram]") {
  bool *d_hit_result = nullptr;
  HitInfo *d_hit_info = nullptr;
  Parallelogram *d_parallelogram = nullptr;
  Ray *d_ray = nullptr;

  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_result, sizeof(bool)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_info, sizeof(HitInfo)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_parallelogram, sizeof(Parallelogram)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_ray, sizeof(Ray)));

  SECTION("Direct hit on XY plane") {
    Vec3 origin(0.0F, 0.0F, 0.0F);
    Vec3 u(1.0F, 0.0F, 0.0F);
    Vec3 v(0.0F, 1.0F, 0.0F);
    *d_parallelogram = Parallelogram(origin, u, v);
    *d_ray = Ray(Vec3(0.5F, 0.5F, 1.0F), Vec3(0.0F, 0.0F, -1.0F));

    testParallelogramHit<<<1, 1>>>(d_parallelogram, d_ray, 0.0F, 100.0F,
                                   d_hit_result, d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    REQUIRE_THAT(d_hit_info->time, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_hit_info->point.x,
                 Catch::Matchers::WithinAbs(0.5F, epsilon));
    REQUIRE_THAT(d_hit_info->point.y,
                 Catch::Matchers::WithinAbs(0.5F, epsilon));
    REQUIRE_THAT(d_hit_info->point.z,
                 Catch::Matchers::WithinAbs(0.0F, epsilon));
  }

  SECTION("Ray misses parallelogram") {
    Vec3 origin(0.0F, 0.0F, 0.0F);
    Vec3 u(1.0F, 0.0F, 0.0F);
    Vec3 v(0.0F, 1.0F, 0.0F);
    *d_parallelogram = Parallelogram(origin, u, v);
    *d_ray = Ray(Vec3(2.0F, 2.0F, 1.0F), Vec3(0.0F, 0.0F, -1.0F));

    testParallelogramHit<<<1, 1>>>(d_parallelogram, d_ray, 0.0F, 100.0F,
                                   d_hit_result, d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == false);
  }

  SECTION("Ray parallel to parallelogram") {
    Vec3 origin(0.0F, 0.0F, 0.0F);
    Vec3 u(1.0F, 0.0F, 0.0F);
    Vec3 v(0.0F, 1.0F, 0.0F);
    *d_parallelogram = Parallelogram(origin, u, v);
    *d_ray = Ray(Vec3(0.5F, 0.5F, 1.0F), Vec3(1.0F, 0.0F, 0.0F));

    testParallelogramHit<<<1, 1>>>(d_parallelogram, d_ray, 0.0F, 100.0F,
                                   d_hit_result, d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == false);
  }

  SECTION("Hit outside time range") {
    Vec3 origin(0.0F, 0.0F, 0.0F);
    Vec3 u(1.0F, 0.0F, 0.0F);
    Vec3 v(0.0F, 1.0F, 0.0F);
    *d_parallelogram = Parallelogram(origin, u, v);
    *d_ray = Ray(Vec3(0.5F, 0.5F, 2.0F), Vec3(0.0F, 0.0F, -1.0F));

    testParallelogramHit<<<1, 1>>>(d_parallelogram, d_ray, 0.0F, 1.0F,
                                   d_hit_result, d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == false);
  }

  CUDA_ERROR_CHECK(cudaFree(d_hit_result));
  CUDA_ERROR_CHECK(cudaFree(d_hit_info));
  CUDA_ERROR_CHECK(cudaFree(d_parallelogram));
  CUDA_ERROR_CHECK(cudaFree(d_ray));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)