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

class Shape {
public:
  __host__ Shape() = default;
  __host__ Shape(const Shape &) = default;
  __host__ Shape(Shape &&) = delete;
  __host__ auto operator=(const Shape &) -> Shape & = default;
  __host__ auto operator=(Shape &&) -> Shape & = delete;
  __host__ virtual ~Shape() = default;
  __host__ __device__ virtual auto hit(const Ray &r) const -> bool = 0;
};
