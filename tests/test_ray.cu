#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {
constexpr auto num_samples = 1024;
constexpr auto seed = 808;

__global__ void benchmarkGetRay(Ray *results, const Vec3 origin,
                                const Vec3 pixel00, const Vec3 deltaU,
                                const Vec3 deltaV, const Vec3 defocusDiskU,
                                const Vec3 defocusDiskV,
                                const float defocusAngle,
                                const int num_samples) {
  curandState_t state;
  curand_init(seed, 0, 0, &state);

  for (int i = 0; i < num_samples; i++) {
    results[i] = getRay(origin, pixel00, deltaU, deltaV, defocusDiskU,
                        defocusDiskV, defocusAngle, 0, 0, state);
  }
}

__global__ void benchmarkGet2Rays(Ray *results, const Vec3 origin,
                                  const Vec3 pixel00, const Vec3 deltaU,
                                  const Vec3 deltaV, const Vec3 defocusDiskU,
                                  const Vec3 defocusDiskV,
                                  const float defocusAngle,
                                  const int num_samples) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, 0, 0, &state);

  for (int i = 0; i < num_samples; i += 2) {
    auto [r1, r2] = get2Rays(origin, pixel00, deltaU, deltaV, defocusDiskU,
                             defocusDiskV, defocusAngle, 0, 0, state);
    results[i] = r1;
    results[i + 1] = r2;
  }
}

__global__ void benchmarkGet4Rays(Ray *results, const Vec3 origin,
                                  const Vec3 pixel00, const Vec3 deltaU,
                                  const Vec3 deltaV, const Vec3 defocusDiskU,
                                  const Vec3 defocusDiskV,
                                  const float defocusAngle,
                                  const int num_samples) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, 0, 0, &state);

  for (int i = 0; i < num_samples; i += 4) {
    auto [r1, r2, r3, r4] =
        get4Rays(origin, pixel00, deltaU, deltaV, defocusDiskU, defocusDiskV,
                 defocusAngle, 0, 0, state);
    results[i] = r1;
    results[i + 1] = r2;
    results[i + 2] = r3;
    results[i + 3] = r4;
  }
}
} // namespace

TEST_CASE("Ray Generation Benchmarks", "[benchmark]") {
  Ray *d_results = nullptr;
  CUDA_ERROR_CHECK(cudaMalloc(&d_results, num_samples * sizeof(Ray)));

  // Setup test parameters
  const Vec3 origin(0.0F, 0.0F, 0.0F);
  const Vec3 pixel00(-1.0F, 1.0F, -1.0F);
  const Vec3 deltaU(2.0F / num_samples, 0.0F, 0.0F);
  const Vec3 deltaV(0.0F, -2.0F / num_samples, 0.0F);
  const Vec3 defocusDiskU(1.0F, 0.0F, 0.0F);
  const Vec3 defocusDiskV(0.0F, 1.0F, 0.0F);
  const float defocusAngle = 0.6F;

  BENCHMARK("Single Ray Generation (curand)") {
    benchmarkGetRay<<<1, 1>>>(d_results, origin, pixel00, deltaU, deltaV,
                              defocusDiskU, defocusDiskV, defocusAngle,
                              num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  BENCHMARK("2 Rays Generation (Philox)") {
    benchmarkGet2Rays<<<1, 1>>>(d_results, origin, pixel00, deltaU, deltaV,
                                defocusDiskU, defocusDiskV, defocusAngle,
                                num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  BENCHMARK("4 Rays Generation (Philox)") {
    benchmarkGet4Rays<<<1, 1>>>(d_results, origin, pixel00, deltaU, deltaV,
                                defocusDiskU, defocusDiskV, defocusAngle,
                                num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  CUDA_ERROR_CHECK(cudaFree(d_results));
}

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

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)