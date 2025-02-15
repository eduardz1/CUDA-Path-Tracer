#include "cuda_path_tracer/materials/dielectric.cuh"

template <typename State>
__device__ auto Dielectric::scatter(const Ray &ray, const Vec3 &normal,
                                    const Vec3 &point, const bool front,
                                    Color &attenuation, Ray &scattered,
                                    State state) const -> bool {
  attenuation = Colors::White;

  const auto ri = front ? (1.0F / refractionIndex) : refractionIndex;
  const auto scatter_direction = makeUnitVector(ray.getDirection());

  const float cos_theta = fmin(dot(-scatter_direction, normal), 1.0F);
  const float sin_theta = sqrtf(1.0F - cos_theta * cos_theta);

  const bool reflecting = ri * sin_theta > 1;
  const auto direction =
      reflecting || reflectance(cos_theta, ri) > curand_uniform(&state)
          ? reflect(scatter_direction, normal)
          : refract(scatter_direction, normal, ri);

  scattered = Ray(point, direction);
  return true;
}

template __device__ auto
Dielectric::scatter<curandState_t>(const Ray &, const Vec3 &, const Vec3 &,
                                   const bool, Color &, Ray &,
                                   curandState_t) const -> bool;
template __device__ auto Dielectric::scatter<curandStatePhilox4_32_10_t>(
    const Ray &, const Vec3 &, const Vec3 &, const bool, Color &, Ray &,
    curandStatePhilox4_32_10_t) const -> bool;

__device__ auto reflect(const Vec3 &v, const Vec3 &n) -> Vec3 {
  return v - 2 * dot(v, n) * n;
}

__device__ auto refract(const Vec3 &v, const Vec3 &n,
                        const float eta_component) -> Vec3 {
  const auto cos_theta = static_cast<float>(std::fmin(dot(-v, n), 1.0));
  const Vec3 r_perp = eta_component * (v + cos_theta * n);
  const Vec3 r_par = -sqrtf(std::fabs(1.0F - r_perp.getLengthSquared())) * n;
  return r_perp + r_par;
}