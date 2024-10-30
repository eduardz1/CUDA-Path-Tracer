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

class ray {
public:
  __host__ __device__ ray();
  __host__ __device__ ray(const vec3 &origin, const vec3 &direction);

  __host__ __device__ auto getOrigin() const -> vec3;
  __host__ __device__ auto getDirection() const -> vec3;

  __host__ __device__ auto at(float t) const -> vec3;

private:
  vec3 origin, direction;
};
