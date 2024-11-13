/**
 * @file sphere.cuh
 * @author Eduard Occhipinti (occhipinti.eduard@icloud.com)
 * @brief Class that represents a sphere in the scene
 * @version 0.1
 * @date 2024-10-30
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "shape.cuh"

class Sphere final : public Shape {
public:
  __host__ __device__ Sphere(const Vec3 &center, float radius);
  __host__ __device__ auto hit(const Ray &r) const -> bool override;

private:
  Vec3 center;
  float radius;
};