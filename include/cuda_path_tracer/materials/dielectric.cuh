#pragma once

#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/ray.cuh"

class Dielectric {
public:
  __host__ __device__ Dielectric(const float refractionIndex)
      : refractionIndex(refractionIndex) {}

  template <typename State>
  __device__ auto scatter(const Ray &ray, const Vec3 &normal, const Vec3 &point,
                          const bool front, Color &attenuation, Ray &scattered,
                          State state) const -> bool;

private:
  /**
   * @brief Refraction index of the material
   */
  float refractionIndex;

  /**
   * @brief Christophe Schlick's approximation for reflectance
   *
   * @param cos cosine of the angle between the ray and the normal
   * @param refraction refraction index
   * @return float reflectance
   */
  __device__ static auto reflectance(float cos,
                                     float refraction_index) -> float {
    const float r0 = (1.0F - refraction_index) / (1.0F + refraction_index);
    return r0 * r0 + (1.0F - r0 * r0) * pow((1.0F - cos), 5.0F); // NOLINT
  }
};

/**
 * @brief Reflect a vector v around a normal n
 *
 * @param v vector to reflect
 * @param n normal
 * @return Vec3 reflected vector
 */
__device__ auto reflect(const Vec3 &v, const Vec3 &n) -> Vec3;

/**
 * @brief Refract a vector v around a normal n
 *
 * @param v vector to refract
 * @param n normal
 * @param eta_component refractive index component
 * @return Vec3 refracted vector
 */
__device__ auto refract(const Vec3 &v, const Vec3 &n,
                        const float eta_component) -> Vec3;