#pragma once

#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/ray.cuh"

/**
 * @brief Metal material, used to simulate metallic materials
 */
class Metal {
public:
  __host__ __device__ Metal(const Color albedo, float fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

  /**
   * @brief Scatter a ray on the material surface and calculate the attenuation
   * and the scattered ray
   *
   * @tparam State type of the curand state
   * @param ray ray that hits the material
   * @param normal normal vector of the material at the hit point
   * @param point hit point
   * @param attenuation attenuation color
   * @param scattered scattered ray
   * @param state curand state
   * @return true if the ray is scattered, false otherwise
   */
  template <typename State>
  __device__ auto scatter(const Ray &ray, const Vec3 &normal, const Vec3 &point,
                          Color &attenuation, Ray &scattered,
                          State &state) const -> bool;

private:
  Color albedo;

  /**
   * @brief Fuzziness of the material, meaning how much the reflection is
   * blurred (0 = no blur, 1 = maximum blur)
   */
  float fuzz;
};
