#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

const auto epsilon = 1e-6F;

__global__ void testCuboidHit(const RectangularCuboid *cuboid, const Ray *ray,
                              float t_min, float t_max, bool *hit_result,
                              HitInfo *hit_info) {
  *hit_result = cuboid->hit(*ray, t_min, t_max, *hit_info);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST_CASE("RectangularCuboid Construction", "[rectangular_cuboid]") {
  SECTION("Basic construction") {
    Vec3 a(0.0F, 0.0F, 0.0F);
    Vec3 b(1.0F, 1.0F, 1.0F);
    REQUIRE_NOTHROW(RectangularCuboid(a, b));
  }

  SECTION("Swapped coordinates") {
    Vec3 a(1.0F, 1.0F, 1.0F);
    Vec3 b(0.0F, 0.0F, 0.0F);
    REQUIRE_NOTHROW(RectangularCuboid(a, b));
  }
}

TEST_CASE("RectangularCuboid Ray Intersection", "[rectangular_cuboid]") {
  bool *d_hit_result = nullptr;
  HitInfo *d_hit_info = nullptr;
  RectangularCuboid *d_cuboid = nullptr;
  Ray *d_ray = nullptr;

  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_result, sizeof(bool)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_info, sizeof(HitInfo)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_cuboid, sizeof(RectangularCuboid)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_ray, sizeof(Ray)));

  SECTION("Front face hit") {
    *d_cuboid = RectangularCuboid(Vec3(0.0F), Vec3(1.0F));
    *d_ray = Ray(Vec3(0.5F, 0.5F, 2.0F), Vec3(0.0F, 0.0F, -1.0F));

    testCuboidHit<<<1, 1>>>(d_cuboid, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    REQUIRE_THAT(d_hit_info->time, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_hit_info->point.z,
                 Catch::Matchers::WithinAbs(1.0F, epsilon));
  }

  SECTION("Miss test") {
    *d_cuboid = RectangularCuboid(Vec3(0.0F), Vec3(1.0F));
    *d_ray = Ray(Vec3(2.0F, 2.0F, 2.0F), Vec3(1.0F, 0.0F, 0.0F));

    testCuboidHit<<<1, 1>>>(d_cuboid, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == false);
  }

  SECTION("Rotation test") {
    *d_cuboid = RectangularCuboid(Vec3(0.0F), Vec3(1.0F));
    d_cuboid->rotate(Vec3(0.0F, 90.0F, 0.0F));
    // After 90-degree Y rotation, the front face is now where +X was
    *d_ray = Ray(Vec3(2.0F, 0.5F, 0.5F), Vec3(-1.0F, 0.0F, 0.0F));

    testCuboidHit<<<1, 1>>>(d_cuboid, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    // The intersection should be at x=1 after rotation
    REQUIRE_THAT(d_hit_info->point.x,
                 Catch::Matchers::WithinAbs(1.0F, epsilon));
    // Normal should point in +X direction after rotation
    REQUIRE_THAT(d_hit_info->normal.x,
                 Catch::Matchers::WithinAbs(1.0F, epsilon));
  }

  SECTION("Translation test") {
    *d_cuboid = RectangularCuboid(Vec3(0.0F), Vec3(1.0F));
    d_cuboid->translate(Vec3(1.0F, 1.0F, 1.0F));
    *d_ray = Ray(Vec3(1.5F, 1.5F, 3.0F), Vec3(0.0F, 0.0F, -1.0F));

    testCuboidHit<<<1, 1>>>(d_cuboid, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    REQUIRE_THAT(d_hit_info->point.z,
                 Catch::Matchers::WithinAbs(2.0F, epsilon));
  }

  SECTION("Combined rotation and translation") {
    *d_cuboid = RectangularCuboid(Vec3(0.0F), Vec3(1.0F));
    d_cuboid->rotate(Vec3(0.0F, 90.0F, 0.0F)).translate(Vec3(1.0F, 1.0F, 1.0F));
    *d_ray = Ray(Vec3(3.0F, 1.5F, 1.5F), Vec3(-1.0F, 0.0F, 0.0F));

    testCuboidHit<<<1, 1>>>(d_cuboid, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    REQUIRE_THAT(d_hit_info->point.x,
                 Catch::Matchers::WithinAbs(2.0F, epsilon));
  }

  CUDA_ERROR_CHECK(cudaFree(d_hit_result));
  CUDA_ERROR_CHECK(cudaFree(d_hit_info));
  CUDA_ERROR_CHECK(cudaFree(d_cuboid));
  CUDA_ERROR_CHECK(cudaFree(d_ray));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)