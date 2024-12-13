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

// enum class ShapeType { SPHERE };

template <class T> class Shape {
public:
  Shape() = default;
  Shape(const Shape &) = default;
  __host__ __device__ Shape(Shape &&) = delete;
  auto operator=(const Shape &) -> Shape & = default;
  __host__ __device__ auto operator=(Shape &&) -> Shape & = delete;
  virtual ~Shape() = default;
  // __host__ __device__ virtual auto hit(const Ray &r) const -> bool = 0;
  __host__ __device__ auto hit(const Ray &r) const -> bool = 0;
  // __host__ __device__ virtual auto getShapeType() const -> ShapeType = 0;
};
