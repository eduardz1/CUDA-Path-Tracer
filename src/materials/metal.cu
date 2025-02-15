#include "cuda_path_tracer/materials/dielectric.cuh"
#include "cuda_path_tracer/materials/metal.cuh"
#include "cuda_path_tracer/vec3.cuh"

template <typename State>
__device__ auto Metal::scatter(const Ray &ray, const Vec3 &normal,
                               const Vec3 &point, Color &attenuation,
                               Ray &scattered, State &state) const -> bool {
  auto reflected_direction = reflect(ray.getDirection(), normal);
  reflected_direction = makeUnitVector(reflected_direction) +
                        (fuzz * randomInUnitDiskRejectionSampling(state));
  scattered = Ray(point, reflected_direction);
  attenuation = albedo;
  return (dot(scattered.getDirection(), normal) > 0);
}

template __device__ auto
Metal::scatter<curandState_t>(const Ray &, const Vec3 &, const Vec3 &, Color &,
                              Ray &, curandState_t &) const -> bool;
template __device__ auto Metal::scatter<curandStatePhilox4_32_10_t>(
    const Ray &, const Vec3 &, const Vec3 &, Color &, Ray &,
    curandStatePhilox4_32_10_t &) const -> bool;
