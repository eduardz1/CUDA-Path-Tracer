#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

constexpr auto epsilon = 1e-6F;
constexpr auto num_samples = 1024;
constexpr auto seed = 808;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {
__global__ void benchmarkRejectionSamplingCurand(Vec3 *result,
                                                 int num_samples) {
  curandState_t state;
  curand_init(seed, 0, 0, &state);
  for (int i = 0; i < num_samples; i++) {
    result[i] = randomInUnitDiskRejectionSampling(state);
  }
}

__global__ void benchmarkRejectionSamplingPhilox(Vec3 *result,
                                                 int num_samples) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, 0, 0, &state);
  for (int i = 0; i < num_samples; i++) {
    result[i] = randomInUnitDiskRejectionSampling(state);
  }
}

__global__ void benchmarkDirect(Vec3 *result, int num_samples) {
  curandStatePhilox4_32_10_t state;
  curand_init(seed, 0, 0, &state);
  for (int i = 0; i < num_samples >> 2; i++) {
    auto [p1, p2, p3, p4] = randomInUnitDisk(state);
    result[i] = p1;
    result[i + 1] = p2;
    result[i + 2] = p3;
    result[i + 3] = p4;
  }
}

__device__ auto isInUnitDisk(const Vec3 &point) -> bool {
  return point.x * point.x + point.y * point.y <= 1.0F && point.z == 0.0F;
}

__global__ void validatePointsInDisk(const Vec3 *points, int n,
                                     int *valid_count) {
  for (int i = 0; i < n; i++) {
    if (isInUnitDisk(points[i])) {
      atomicAdd(valid_count, 1);
    }
  }
}
} // namespace

// FIXME: These benchmarks use CPU Timers, which are not ideal because they
// stall the GPU pipeline, I haven't found a way to integrate cudaEvents with
// Catch2's benchmarking framework, but, eventually, they would be the better
// option.
TEST_CASE("Random Unit Disk Generation Benchmarks", "[benchmark]") {
  Vec3 *d_result = nullptr;
  CUDA_ERROR_CHECK(cudaMalloc(&d_result, num_samples * sizeof(Vec3)));

  BENCHMARK("Rejection Sampling (curand)") {
    benchmarkRejectionSamplingCurand<<<1, 1>>>(d_result, num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  BENCHMARK("Rejection Sampling (Philox)") {
    benchmarkRejectionSamplingPhilox<<<1, 1>>>(d_result, num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  BENCHMARK("Direct Generation (Philox)") {
    benchmarkDirect<<<1, 1>>>(d_result, num_samples);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  };

  CUDA_ERROR_CHECK(cudaFree(d_result));
}

TEST_CASE("Random In Unit Disk Tests", "[random]") {
  const int num_points = 10000;
  Vec3 *d_points = nullptr;
  int *d_valid_count = nullptr;

  CUDA_ERROR_CHECK(cudaMalloc(&d_points, num_points * sizeof(Vec3)));
  CUDA_ERROR_CHECK(cudaMalloc(&d_valid_count, sizeof(int)));

  SECTION("Rejection Sampling (curand)") {
    CUDA_ERROR_CHECK(cudaGetLastError());
    benchmarkRejectionSamplingCurand<<<1, 1>>>(d_points, num_points);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Check all points are in disk
    CUDA_ERROR_CHECK(cudaMemset(d_valid_count, 0, sizeof(int)));
    validatePointsInDisk<<<1, 1>>>(d_points, num_points, d_valid_count);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    int valid_count = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&valid_count, d_valid_count, sizeof(int),
                                cudaMemcpyDeviceToHost));
    REQUIRE(valid_count == num_points);
  }

  SECTION("Rejection Sampling (Philox)") {
    CUDA_ERROR_CHECK(cudaGetLastError());
    benchmarkRejectionSamplingPhilox<<<1, 1>>>(d_points, num_points);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    CUDA_ERROR_CHECK(cudaMemset(d_valid_count, 0, sizeof(int)));
    validatePointsInDisk<<<1, 1>>>(d_points, num_points, d_valid_count);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    int valid_count = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&valid_count, d_valid_count, sizeof(int),
                                cudaMemcpyDeviceToHost));
    REQUIRE(valid_count == num_points);
  }

  SECTION("Direct Generation (Philox)") {
    CUDA_ERROR_CHECK(cudaGetLastError());
    benchmarkDirect<<<1, 1>>>(d_points, num_points);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    CUDA_ERROR_CHECK(cudaMemset(d_valid_count, 0, sizeof(int)));
    validatePointsInDisk<<<1, 1>>>(d_points, num_points, d_valid_count);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    int valid_count = 0;
    CUDA_ERROR_CHECK(cudaMemcpy(&valid_count, d_valid_count, sizeof(int),
                                cudaMemcpyDeviceToHost));
    REQUIRE(valid_count == num_points);

    // Additional test for z-coordinate being 0
    std::vector<Vec3> host_points(num_points);
    CUDA_ERROR_CHECK(cudaMemcpy(host_points.data(), d_points,
                                num_points * sizeof(Vec3),
                                cudaMemcpyDeviceToHost));

    for (const auto &point : host_points) {
      REQUIRE_THAT(point.z, Catch::Matchers::WithinAbs(0.0F, epsilon));
      REQUIRE(point.x * point.x + point.y * point.y <= 1.0F + epsilon);
    }
  }

  CUDA_ERROR_CHECK(cudaFree(d_points));
  CUDA_ERROR_CHECK(cudaFree(d_valid_count));
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)