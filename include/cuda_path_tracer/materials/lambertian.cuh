#pragma once

#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/texture.cuh"

/**
 * @brief Lambertian material, used to simulate diffuse materials
 */
class Lambertian {

public:
  __host__ __device__ Lambertian(const Texture &texture) : texture(texture) {}

  /**
   * @brief Scatter a ray on the material surface and calculate the attenuation
   * color and the scattered ray
   *
   * @tparam State type of the curand state
   * @param normal normal vector of the material at the hit point
   * @param point hit point
   * @param attenuation attenuation color
   * @param scattered scattered ray
   * @param state curand state
   * @return true if the ray is scattered, false otherwise
   */
  template <typename State>
  __device__ auto scatter(const Vec3 &normal, const Vec3 &point,
                          Color &attenuation, Ray &scattered,
                          State &state) const -> bool;

private:
  Texture texture;
};
