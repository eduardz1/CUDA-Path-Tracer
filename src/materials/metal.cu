#include "cuda_path_tracer/materials/metal.cuh"

template <typename State>
__device__ auto Metal::scatter(const Ray &ray, const Vec3 &normal,
                               const Vec3 &point, Vec3 &attenuation,
                               Ray &scattered, State &state) const -> bool {
  auto reflected_direction = reflect(ray.getDirection(), normal);
  reflected_direction = makeUnitVector(reflected_direction) +
                        (fuzz * makeUnitVector(randomVector(state)));
  scattered = Ray(point, reflected_direction);
  attenuation = albedo;
  return (dot(scattered.getDirection(), normal) > 0);
}

template __device__ auto
Metal::scatter<curandState_t>(const Ray &, const Vec3 &, const Vec3 &, Vec3 &,
                              Ray &, curandState_t &) const -> bool;
template __device__ auto Metal::scatter<curandStatePhilox4_32_10_t>(
    const Ray &, const Vec3 &, const Vec3 &, Vec3 &, Ray &,
    curandStatePhilox4_32_10_t &) const -> bool;

__device__ auto Metal::emitted(Vec3 &point) -> Vec3 { return Vec3{0, 0, 0}; }
