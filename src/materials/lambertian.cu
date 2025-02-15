#include "cuda_path_tracer/materials/lambertian.cuh"
#include "cuda_path_tracer/utilities.cuh"

template <typename State>
__device__ auto Lambertian::scatter(const Vec3 &normal, const Vec3 &point,
                                    Vec3 &attenuation, Ray &scattered,
                                    State &state) const -> bool {
  auto scatter_direction = normal + vectorOnHemisphere<State>(normal, state);
  scatter_direction = roundScatterDirection(scatter_direction, normal);
  scattered = Ray(point, scatter_direction);
  attenuation =
      cuda::std::visit(overload{[&point](const Checker &checker) {
                                  return checker.texture_value(point);
                                },
                                [](const Color &color) { return Vec3{color}; }},
                       texture);
  return true;
}

template __device__ auto
Lambertian::scatter<curandState_t>(const Vec3 &, const Vec3 &, Vec3 &, Ray &,
                                   curandState_t &) const -> bool;
template __device__ auto Lambertian::scatter<curandStatePhilox4_32_10_t>(
    const Vec3 &, const Vec3 &, Vec3 &, Ray &,
    curandStatePhilox4_32_10_t &) const -> bool;
