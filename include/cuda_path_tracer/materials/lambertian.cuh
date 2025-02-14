#pragma once

#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/texture.cuh"

class Lambertian {

public:
  __host__ __device__ Lambertian() : texture(Solid(Vec3{0})) {}
  __host__ __device__ Lambertian(const Texture &texture) : texture(texture) {}
  __host__ __device__ Lambertian(const Vec3 albedo) : texture(Solid(albedo)) {}

  template <typename State>
  __device__ auto scatter(const Vec3 &normal, const Vec3 &point,
                          Vec3 &attenuation, Ray &scattered, State &state) const
      -> bool;

  __device__ auto emitted(Vec3 &point) -> Vec3;

private:
  Texture texture;
};
