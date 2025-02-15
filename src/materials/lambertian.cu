#include "cuda_path_tracer/materials/lambertian.cuh"
#include "cuda_path_tracer/utilities.cuh"
#include "cuda_path_tracer/vec3.cuh"

template <typename State>
__device__ auto Lambertian::scatter(const Vec3 &normal, const Vec3 &point,
                                    Color &attenuation, Ray &scattered,
                                    State &state) const -> bool {
  // auto scatter_direction = normal + vectorOnHemisphere<State>(normal, state);
  // scatter_direction = roundScatterDirection(scatter_direction, normal);
  auto scatter_direction = normal + randomInUnitDiskRejectionSampling(state);

  if (scatter_direction.nearZero()) {
    scatter_direction = normal;
  }

  scattered = Ray(point, scatter_direction);
  attenuation =
      cuda::std::visit(overload{[&point](const Checker &checker) {
                                  return checker.texture_value(point);
                                },
                                [](const Color &color) { return color; }},
                       texture);
  return true;
}

template __device__ auto
Lambertian::scatter<curandState_t>(const Vec3 &, const Vec3 &, Color &, Ray &,
                                   curandState_t &) const -> bool;
template __device__ auto Lambertian::scatter<curandStatePhilox4_32_10_t>(
    const Vec3 &, const Vec3 &, Color &, Ray &,
    curandStatePhilox4_32_10_t &) const -> bool;
