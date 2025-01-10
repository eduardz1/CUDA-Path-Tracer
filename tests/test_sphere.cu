#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

const auto epsilon = 1e-6F;

__global__ void testGetCenter(const Sphere *sphere, Vec3 *result) {
  *result = sphere->getCenter();
}

__global__ void testSphereHit(const Sphere *sphere, const Ray *ray, float t_min,
                              float t_max, bool *hit_result,
                              HitInfo *hit_info) {
  *hit_result = sphere->hit(*ray, t_min, t_max, *hit_info);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST_CASE("Sphere construction and basic properties", "[sphere]") {
  Vec3 *d_center = nullptr;
  Sphere *d_sphere = nullptr;

  CUDA_ERROR_CHECK(cudaMallocManaged(&d_center, sizeof(Vec3)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_sphere, sizeof(Sphere)));

  Vec3 center(1.0F, 2.0F, 3.0F);
  float radius = 2.0F;

  *d_sphere = Sphere(center, radius);
  testGetCenter<<<1, 1>>>(d_sphere, d_center);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  REQUIRE(d_center->x == center.x);
  REQUIRE(d_center->y == center.y);
  REQUIRE(d_center->z == center.z);

  CUDA_ERROR_CHECK(cudaFree(d_center));
  CUDA_ERROR_CHECK(cudaFree(d_sphere));
}

TEST_CASE("Sphere ray intersection tests", "[sphere]") {
  bool *d_hit_result = nullptr;
  HitInfo *d_hit_info = nullptr;
  Sphere *d_sphere = nullptr;
  Ray *d_ray = nullptr;

  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_result, sizeof(bool)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_hit_info, sizeof(HitInfo)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_sphere, sizeof(Sphere)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_ray, sizeof(Ray)));

  SECTION("Direct hit") {
    *d_sphere = Sphere(Vec3(0, 0, 0), 1.0F);
    *d_ray = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1));

    testSphereHit<<<1, 1>>>(d_sphere, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
    REQUIRE_THAT(d_hit_info->time, Catch::Matchers::WithinAbs(4.0F, epsilon));
  }

  SECTION("Ray miss") {
    *d_sphere = Sphere(Vec3(0, 0, 0), 1.0F);
    *d_ray = Ray(Vec3(2, 2, -5), Vec3(0, 0, 1));

    testSphereHit<<<1, 1>>>(d_sphere, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == false);
  }

  SECTION("Ray from inside") {
    *d_sphere = Sphere(Vec3(0, 0, 0), 2.0F);
    *d_ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1));

    testSphereHit<<<1, 1>>>(d_sphere, d_ray, 0.0F, 100.0F, d_hit_result,
                            d_hit_info);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    REQUIRE(*d_hit_result == true);
  }

  CUDA_ERROR_CHECK(cudaFree(d_hit_result));
  CUDA_ERROR_CHECK(cudaFree(d_hit_info));
  CUDA_ERROR_CHECK(cudaFree(d_sphere));
  CUDA_ERROR_CHECK(cudaFree(d_ray));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)