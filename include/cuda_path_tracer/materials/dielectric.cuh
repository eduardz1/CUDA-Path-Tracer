#pragma once

#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/ray.cuh"

/**
 * @brief Dielectric material, used to simulate transparent materials
 */
class Dielectric {
public:
  __host__ __device__ Dielectric(const float refractionIndex)
      : refractionIndex(refractionIndex) {}

  /**
   * @brief Scatter a ray on the material surface and calculate the attenuation
   * color and the scattered ray
   *
   * @tparam State type of the curand state
   * @param ray ray that hits the material
   * @param normal normal vector of the material at the hit point
   * @param point hit point
   * @param front true if the ray hits the front side of the material, false if
   * it hits the back side
   * @param attenuation attenuation color
   * @param scattered scattered ray
   * @param state curand state
   * @return true if the ray is scattered, false otherwise
   */
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