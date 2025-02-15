#pragma once

#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/ray.cuh"

class Metal {
public:
  __host__ __device__ Metal(const Color albedo, float fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

  template <typename State>
  __device__ auto scatter(const Ray &ray, const Vec3 &normal, const Vec3 &point,
                          Color &attenuation, Ray &scattered,
                          State &state) const -> bool;

private:
  Color albedo;
  float fuzz;
};
