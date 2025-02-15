#pragma once

#include <curand_kernel.h>
#include <ostream>

/**
 * @brief Class for a 3D vector
 *
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

  /**
   * @brief Enables implicit conversion to the float4 type, useful for image
   * processing with CUDA and particularly in the conversion to uchar4 type. By
   * default the last float s assigned to 1.0f (alpha channel in case of an
   * image).
   *
   * @return float4
   */
  __device__ operator float4() const;

  __host__ __device__ auto getLengthSquared() const -> float;
  __host__ __device__ auto getLength() const -> float;
};

__host__ auto operator<<(std::ostream &os, const Vec3 &v) -> std::ostream &;
__host__ __device__ auto operator+(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator-(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator*(const Vec3 &v, const float t) -> Vec3;
__host__ __device__ auto operator*(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto operator/(const Vec3 &v, float t) -> Vec3;

__host__ __device__ auto cross(const Vec3 &v1, const Vec3 &v2) -> Vec3;
__host__ __device__ auto dot(const Vec3 &v1, const Vec3 &v2) -> float;

__host__ __device__ auto makeUnitVector(const Vec3 &v) -> Vec3;

template <typename State>
__device__ auto vectorOnHemisphere(const Vec3 &v, State &state) -> Vec3;

__device__ auto randomVector(curandStatePhilox4_32_10_t &state) -> Vec3;
__device__ auto randomVector(curandState_t &state) -> Vec3;

__device__ auto roundScatterDirection(const Vec3 &direction,
                                      const Vec3 &normal) -> Vec3;
__device__ auto reflect(const Vec3 &v, const Vec3 &n) -> Vec3;
__device__ auto refract(const Vec3 &v, const Vec3 &n,
                        float eta_component) -> Vec3;