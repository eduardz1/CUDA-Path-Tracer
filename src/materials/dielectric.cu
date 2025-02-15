#include "cuda_path_tracer/materials/dielectric.cuh"

__device__ auto Dielectric::scatter(const Ray &ray, const Vec3 &normal,
                                    const Vec3 &point, const bool front,
                                    Vec3 &attenuation, Ray &scattered) const
    -> bool {
  const auto ri = front ? (1.0F / refraction) : refraction;
  const auto scatter_direction = makeUnitVector(ray.getDirection());

  const float cos_theta = fmin(dot(-scatter_direction, normal), 1.0F);
  const float sin_theta = sqrtf(1.0F - cos_theta * cos_theta);
  const bool reflecting = ri * sin_theta > 1;

  scattered = Ray(point, reflecting ? reflect(scatter_direction, normal)
                                    : refract(scatter_direction, normal, ri));
  attenuation = Vec3{1.0F, 1.0F, 1.0F};
  return true;
}
