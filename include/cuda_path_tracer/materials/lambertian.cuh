#pragma once

#include "cuda_path_tracer/ray.cuh"

class Lambertian {

public:
  __host__ __device__ Lambertian(const Vec3 albedo) : albedo(albedo) {}

  template <typename State>
  __device__ auto scatter(const Vec3 &normal,const  Vec3 &point, Vec3 &attenuation,
                          Ray &scattered, State &state) const -> bool;

private:
  Vec3 albedo;
};
