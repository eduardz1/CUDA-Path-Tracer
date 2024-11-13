/**
 * @file shape.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Abstract class for a shape in the scene
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "cuda_path_tracer/ray.cuh"

extern "C" __device__ void __cxa_pure_virtual() { // NOLINT
  while (1) {
  }
}
class Shape {
public:
  __host__ __device__ Shape() = default;
  __host__ __device__ Shape(const Shape &) = default;
  __host__ __device__ Shape(Shape &&) = delete;
  __host__ __device__ auto operator=(const Shape &) -> Shape & = default;
  __host__ __device__ auto operator=(Shape &&) -> Shape & = delete;
  __host__ __device__ virtual ~Shape() = default;
  __host__ __device__ virtual auto hit(const Ray &r) const -> bool = 0;
};
