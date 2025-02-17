#pragma once

#include <cuda/std/tuple>
#include <curand_kernel.h>
#include <ostream>

/**
 * @brief Class for a 3D vector
 */
struct Vec3 {
  float x, y, z;

  __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ constexpr Vec3(const float value)
      : x(value), y(value), z(value) {}
  __host__ __device__ constexpr Vec3(const float x_, const float y_,
                                     const float z_)
      : x(x_), y(y_), z(z_) {}
  __host__ __device__ constexpr Vec3(const float4 &v)
      : x(v.x), y(v.y), z(v.z) {}

  __host__ __device__ auto operator-() const -> Vec3;
  __host__ __device__ auto operator==(const Vec3 &v) const -> bool;
  __host__ __device__ auto operator+=(const Vec3 &v) -> Vec3 &;
  __host__ __device__ auto operator*=(const Vec3 &v) -> Vec3 &;
  __host__ __device__ auto operator/=(const float t) -> Vec3 &;

  /**
   * @brief Enables implicit conversion to the float4 type, useful for image
   * processing with CUDA and particularly in the conversion to uchar4 type. By
   * default the last float s assigned to 1.0f (alpha channel in case of an
   * image).
   *
   * @return float4
   */
  __device__ operator float4() const;

  /**
   * @brief Get the squared length of the vector
   *
   * @return float The squared length of the vector
   */
  __host__ __device__ auto getLengthSquared() const -> float;

  /**
   * @brief Get the length of the vector
   *
   * @return float The length of the vector
   */
  __host__ __device__ auto getLength() const -> float;

  /**
   * @brief Check if the vector is near zero
   *
   * @return true if the vector is near zero
   * @return false if the vector is not near zero
   */
  __host__ __device__ auto nearZero() const -> bool;
};

__host__ auto operator<<(std::ostream &os, const Vec3 &v) -> std::ostream &;
__host__ __device__ auto operator+(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator-(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator*(const Vec3 &v, const float t) -> Vec3;
__host__ __device__ auto operator*(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator/(const Vec3 &v, float t) -> Vec3;

/**
 * @brief Calculate the cross product of two vectors
 *
 * @param v1 The first vector
 * @param v2 The second vector
 * @return Vec3 The cross product
 */
__host__ __device__ auto cross(const Vec3 &v1, const Vec3 &v2) -> Vec3;

/**
 * @brief Calculate the dot product of two vectors
 *
 * @param v1 The first vector
 * @param v2 The second vector
 * @return float The dot product
 */
__host__ __device__ auto dot(const Vec3 &v1, const Vec3 &v2) -> float;

/**
 * @brief Create a unit vector from a given vector
 *
 * @param v The vector
 * @return Vec3 The unit vector
 */
__host__ __device__ auto makeUnitVector(const Vec3 &v) -> Vec3;

/**
 * @brief Generate a random point in a unit disk through rejection sampling,
 * meaning that we will keep generating random points until we find one that is
 * within the unit disk.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit disk
 */
__device__ auto randomInUnitDiskRejectionSampling(curandState_t &state) -> Vec3;

/**
 * @brief Generate a random point in a unit disk through rejection sampling but
 * two choices at a time to reduce the number of iterations from pi/4 to pi/2 on
 * average. Furthermore, a single call of `curand_uniform4` is used to generate
 * four random numbers at once instead of two calls to `curand_uniform` like in
 * the `curandState_t` version.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit disk
 */
__device__ auto
randomInUnitDiskRejectionSampling(curandStatePhilox4_32_10_t &state) -> Vec3;

/**
 * @brief Generates four random points at a time in a unit disk by extrapolating
 * the density function of the disk.
 *
 * @param state The curand state
 * @return cuda::std::tuple<Vec3, Vec3, Vec3, Vec3> The four random points in
 * the unit disk
 */
__device__ auto randomInUnitDisk(curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3>;

/**
 * @brief Generate a random point in a unit sphere through rejection sampling,
 * meaning that we will keep generating random points until we find one that is
 * within the unit sphere.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit sphere
 */
__device__ auto
randomInUnitSphereRejectionSampling(curandStatePhilox4_32_10_t &state) -> Vec3;

/**
 * @brief Generate a random point in a unit sphere through rejection sampling,
 * meaning that we will keep generating random points until we find one that is
 * within the unit sphere.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit sphere
 */
__device__ auto
randomInUnitSphereRejectionSampling(curandState_t &state) -> Vec3;
