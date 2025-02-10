#include "catch2/matchers/catch_matchers.hpp"
#include "cuda_path_tracer/error.cuh"
#define protected public
#include "cuda_path_tracer/shapes/rotation.cuh"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

const auto epsilon = 1e-6F;

__global__ void testRotatePoint(const Rotation *rotation, Vec3 *point,
                                Vec3 *result, bool inverse) {
  *result = rotation->rotate(*point, inverse);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST_CASE("Rotation Construction and Caching", "[rotation]") {

  SECTION("Zero rotation") {
    Rotation rot(Vec3(0.0F, 0.0F, 0.0F));
    Vec3 *d_point = nullptr;
    Vec3 *d_result = nullptr;
    Rotation *d_rotation = nullptr;

    CUDA_ERROR_CHECK(cudaMallocManaged(&d_point, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_result, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_rotation, sizeof(Rotation)));

    *d_point = Vec3(1.0F, 0.0F, 0.0F);
    *d_rotation = rot;

    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, false);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    REQUIRE_THAT(d_result->x, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_result->y, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->z, Catch::Matchers::WithinAbs(0.0F, epsilon));

    CUDA_ERROR_CHECK(cudaFree(d_point));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    CUDA_ERROR_CHECK(cudaFree(d_rotation));
  }

  SECTION("90 degree X rotation") {
    Rotation rot(Vec3(90.0F, 0.0F, 0.0F));
    Vec3 *d_point = nullptr;
    Vec3 *d_result = nullptr;
    Rotation *d_rotation = nullptr;

    CUDA_ERROR_CHECK(cudaMallocManaged(&d_point, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_result, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_rotation, sizeof(Rotation)));

    *d_point = Vec3(0.0F, 1.0F, 0.0F);
    *d_rotation = rot;

    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, false);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    REQUIRE_THAT(d_result->x, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->y, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->z, Catch::Matchers::WithinAbs(1.0F, epsilon));

    CUDA_ERROR_CHECK(cudaFree(d_point));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    CUDA_ERROR_CHECK(cudaFree(d_rotation));
  }

  SECTION("Combined rotation") {
    Rotation rot(Vec3(90.0F, 90.0F, 90.0F));
    Vec3 point(0.0F, 1.0F, 0.0F);
    Vec3 *d_point = nullptr;
    Vec3 *d_result = nullptr;
    Rotation *d_rotation = nullptr;

    CUDA_ERROR_CHECK(cudaMallocManaged(&d_point, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_result, sizeof(Vec3)));
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_rotation, sizeof(Rotation)));

    CUDA_ERROR_CHECK(
        cudaMemcpy(d_point, &point, sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_rotation, &rot, sizeof(Rotation), cudaMemcpyHostToDevice));

    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, false);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Values for 90-degree rotation on all axes
    REQUIRE_THAT(d_result->x, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->y, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_result->z, Catch::Matchers::WithinAbs(0.0F, epsilon));

    CUDA_ERROR_CHECK(cudaFree(d_point));
    CUDA_ERROR_CHECK(cudaFree(d_result));
    CUDA_ERROR_CHECK(cudaFree(d_rotation));
  }
}

// FIXME: Inverse rotation is broken, maybe it's a problem with the code
TEST_CASE("Inverse Rotation Tests", "[rotation]") {
  Vec3 *d_point = nullptr;
  Vec3 *d_result = nullptr;
  Vec3 *d_final = nullptr;
  Rotation *d_rotation = nullptr;

  CUDA_ERROR_CHECK(cudaMallocManaged(&d_point, sizeof(Vec3)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_result, sizeof(Vec3)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_final, sizeof(Vec3)));
  CUDA_ERROR_CHECK(cudaMallocManaged(&d_rotation, sizeof(Rotation)));

  SECTION("Inverse X rotation 90 degrees") {
    Rotation rot(Vec3(90.0F, 0.0F, 0.0F));
    Vec3 point(0.0F, 1.0F, 0.0F);
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_point, &point, sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_rotation, &rot, sizeof(Rotation), cudaMemcpyHostToDevice));

    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, true);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    REQUIRE_THAT(d_result->x, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->y, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->z, Catch::Matchers::WithinAbs(-1.0F, epsilon));
  }

  SECTION("Rotation followed by inverse returns original point") {
    Rotation rot(Vec3(45.0F, 30.0F, 60.0F));
    Vec3 point(1.0F, 0.0F, 0.0F);
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_point, &point, sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_rotation, &rot, sizeof(Rotation), cudaMemcpyHostToDevice));

    // Forward rotation
    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, false);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Inverse rotation
    testRotatePoint<<<1, 1>>>(d_rotation, d_result, d_final, true); // NOLINT
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    REQUIRE_THAT(d_final->x, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_final->y, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_final->z, Catch::Matchers::WithinAbs(0.0F, epsilon));
  }

  SECTION("Inverse combined rotation") {
    Rotation rot(Vec3(90.0F, 90.0F, 90.0F));
    Vec3 point(0.0F, 1.0F, 0.0F);

    CUDA_ERROR_CHECK(
        cudaMemcpy(d_point, &point, sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(
        cudaMemcpy(d_rotation, &rot, sizeof(Rotation), cudaMemcpyHostToDevice));

    testRotatePoint<<<1, 1>>>(d_rotation, d_point, d_result, true);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaGetLastError());

    REQUIRE_THAT(d_result->x, Catch::Matchers::WithinAbs(0.0F, epsilon));
    REQUIRE_THAT(d_result->y, Catch::Matchers::WithinAbs(1.0F, epsilon));
    REQUIRE_THAT(d_result->z, Catch::Matchers::WithinAbs(0.0F, epsilon));
  }

  CUDA_ERROR_CHECK(cudaFree(d_point));
  CUDA_ERROR_CHECK(cudaFree(d_result));
  CUDA_ERROR_CHECK(cudaFree(d_final));
  CUDA_ERROR_CHECK(cudaFree(d_rotation));
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)