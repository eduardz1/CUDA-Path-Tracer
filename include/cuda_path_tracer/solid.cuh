#pragma once
#include "vec3.cuh"

class Solid {

public:
  __host__ __device__ Solid(const Vec3 albedo) : albedo(albedo) {}

  __device__ Vec3 texture_value(Vec3 &point) {
    return albedo;
  }

private:
  Vec3 albedo;
};
