#include "cuda_path_tracer/materials/lambertian.cuh"

template <typename State>
__device__ auto Lambertian::scatter(const Vec3 &normal, const Vec3 &point,
                                    Vec3 &attenuation, Ray &scattered,
                                    State &state) const -> bool {
  auto scatter_direction = normal + vectorOnHemisphere<State>(normal, state);
  scatter_direction = roundScatterDirection(scatter_direction, normal);
  scattered = Ray(point, scatter_direction);
  attenuation = albedo;
  return true;
}

template __device__ auto
Lambertian::scatter<curandState_t>(const Vec3 &, const Vec3 &, Vec3 &, Ray &,
                                   curandState_t &) const -> bool;
template __device__ auto Lambertian::scatter<curandStatePhilox4_32_10_t>(
    const Vec3 &, const Vec3 &, Vec3 &, Ray &,
    curandStatePhilox4_32_10_t &) const -> bool;