/**
 * @file ray.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Header file for ray.cu, which contains the ray class as described in
 * the book "Ray Tracing Gems", published by NVIDIA
 * @version 0.1
 * @date 2024-10-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>

class Ray {
public:
  __host__ __device__ Ray();
  __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction);

  __host__ __device__ auto getOrigin() const -> Vec3;
  __host__ __device__ auto getDirection() const -> Vec3;

  /**
   * @brief Returns the point at a given t value
   *
   * @param t time value
   * @return Vec3 point at time t
   */
  __host__ __device__ auto at(float t) const -> Vec3;

private:
  Vec3 origin, direction;
};
