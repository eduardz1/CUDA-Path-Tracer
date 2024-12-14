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

#include "ray.cuh"
#include "vec3.cuh"

class Sphere {
public:
  __host__ Sphere(const Vec3 &center, float radius);
  __device__ auto hit(const Ray &r) const -> bool;

private:
  Vec3 center;
  float radius;
};