/**
 * @file vec3.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Header file for vec3.cu, which contains the vec3 class
 * @version 0.1
 * @date 2024-10-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <driver_types.h>
#include <ostream>

/**
 * @brief Class for a 3D vector
 *
 */
class Vec3 {
public:
  __host__ __device__ Vec3();
  __host__ __device__ Vec3(float x);
  __host__ __device__ Vec3(float x, float y, float z);

  __host__ __device__ auto getX() const -> float;
  __host__ __device__ auto getY() const -> float;
  __host__ __device__ auto getZ() const -> float;

  __host__ __device__ auto operator+(const Vec3 &v) const -> Vec3;
  __host__ __device__ auto operator-(const Vec3 &v) const -> Vec3;
  __host__ __device__ auto operator/(const Vec3 &v) const -> Vec3;
  __host__ __device__ auto operator==(const Vec3 &v) const -> bool;

  __host__ __device__ auto dot(const Vec3 &v) const -> float;

private:
  float x, y, z;
};

__host__ auto operator<<(std::ostream &os, const Vec3 &v) -> std::ostream &;
__host__ __device__ auto operator*(float t, const Vec3 &v) -> Vec3;
__host__ __device__ auto operator*(const Vec3 &v, float t) -> Vec3;