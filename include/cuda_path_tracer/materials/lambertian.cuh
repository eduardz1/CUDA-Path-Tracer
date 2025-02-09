#pragma once

#include "cuda_path_tracer/ray.cuh"

class Lambertian {

public:
  __host__ __device__ Lambertian(const Vec3 albedo) : albedo(albedo) {}

  __device__ auto scatter(const Ray &ray, Vec3 &normal, Vec3 &point, bool front,
                          Vec3 &attenuation, Ray &scattered,
                          curandStatePhilox4_32_10_t &state) -> bool {
    auto scatter_direction = normal + vectorOnHemisphere(normal, state);
    scatter_direction = roundScatterDirection(scatter_direction, normal);
    scattered = Ray(point, scatter_direction);
    attenuation = albedo;
    return true;
  }

private:
  Vec3 albedo;
};
