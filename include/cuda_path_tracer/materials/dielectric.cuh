#pragma once

#include "cuda_path_tracer/ray.cuh"
class Dielectric {

public:
  __host__ __device__ Dielectric(const float refraction)
      : refraction(refraction) {}

  __device__ auto scatter(const Ray &ray, const Vec3 &normal, const Vec3 &point,
                          const bool front, Vec3 &attenuation,
                          Ray &scattered) const -> bool;

private:
  float refraction;

  static auto reflectance(float cos, float refraction) -> float {
    const float r0 = (1.0F - refraction) / (1.0F + refraction);
    return r0 * r0 + (1.0F - r0 * r0) * pow((1.0F - cos), 5.0F);
  }
};
