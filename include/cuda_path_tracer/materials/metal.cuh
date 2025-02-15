#pragma once

#include "cuda_path_tracer/ray.cuh"

class Metal {
public:
  __host__ __device__ Metal(const Vec3 albedo, float fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

  template <typename State>
  __device__ auto scatter(const Ray &ray, const Vec3 &normal, const Vec3 &point,
                          Vec3 &attenuation, Ray &scattered,
                          State &state) const -> bool;

private:
  Vec3 albedo;
  float fuzz;
};
