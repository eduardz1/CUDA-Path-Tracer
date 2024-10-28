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

#include <__clang_cuda_runtime_wrapper.h>
#include <driver_types.h>

/**
 * @brief Class for a 3D vector
 *
 */
class vec3 {
public:
  __host__ __device__ vec3();
  __host__ __device__ vec3(float x);
  __host__ __device__ vec3(float x, float y, float z);

  __host__ __device__ [[nodiscard]] auto getX() const -> float;
  __host__ __device__ [[nodiscard]] auto getY() const -> float;
  __host__ __device__ [[nodiscard]] auto getZ() const -> float;

  __host__ __device__ auto operator+(const vec3 &v) const -> vec3;
  __host__ __device__ auto operator-(const vec3 &v) const -> vec3;
  __host__ __device__ auto operator*(const vec3 &v) const -> vec3;
  __host__ __device__ auto operator*(float t) const -> vec3;
  __host__ __device__ auto operator/(const vec3 &v) const -> vec3;
  __host__ __device__ auto operator==(const vec3 &v) const -> bool;

  __host__ __device__ [[nodiscard]] auto dot(const vec3 &v) const -> float;

private:
  float x, y, z;
};