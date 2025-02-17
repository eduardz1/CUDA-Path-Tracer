#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/tuple>

/**
 * @brief Ray class, used to represent a ray in 3D space
 */
class Ray {
public:
  __host__ __device__ Ray();
  __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction);

  __host__ __device__ auto getOrigin() const -> Vec3;
  __host__ __device__ auto getDirection() const -> Vec3;

  /**
   * @brief Returns the point at a given t value
   *
   * @param t time value
   * @return Vec3 point at time t
   */
  __host__ __device__ auto at(float t) const -> Vec3;

private:
  Vec3 origin, direction;
};

// Functions to generate rays

/**
 * @brief Calculate the ray for a given pixel and offset it by a random value
 * for antialiasing
 *
 * @param origin origin of the ray
 * @param pixel00 pixel at the bottom left corner of the screen
 * @param deltaU vector representing the change in the x direction
 * @param deltaV vector representing the change in the y direction
 * @param defocusDiskU vector representing the defocus disk in the x direction
 * @param defocusDiskV vector representing the defocus disk in the y direction
 * @param defocusAngle angle of the defocus disk
 * @param x x coordinate of the pixel
 * @param y y coordinate of the pixel
 * @param state curand state
 * @return Ray generated ray
 */
__device__ auto getRay(const Vec3 &origin, const Vec3 &pixel00,
                       const Vec3 &deltaU, const Vec3 &deltaV,
                       const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                       const float defocusAngle, const uint16_t x,
                       const uint16_t y, curandState_t &state) -> Ray;

/**
 * @brief Calculate two rays for a given pixel and offset them by a random value
 * for antialiasing
 *
 * @param origin origin of the ray
 * @param pixel00 pixel at the bottom left corner of the screen
 * @param deltaU vector representing the change in the x direction
 * @param deltaV vector representing the change in the y direction
 * @param defocusDiskU vector representing the defocus disk in the x direction
 * @param defocusDiskV vector representing the defocus disk in the y direction
 * @param defocusAngle angle of the defocus disk
 * @param x x coordinate of the pixel
 * @param y y coordinate of the pixel
 * @param state curand state
 * @return cuda::std::tuple<Ray, Ray> generated rays
 */
__device__ auto
get2Rays(const Vec3 &origin, const Vec3 &pixel00, const Vec3 &deltaU,
         const Vec3 &deltaV, const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
         const float defocusAngle, const uint16_t x, const uint16_t y,
         curandStatePhilox4_32_10_t &state) -> cuda::std::tuple<Ray, Ray>;

/**
 * @brief Calculate four rays for a given pixel and offset them by a random
 * value for antialiasing
 *
 * @param origin origin of the ray
 * @param pixel00 pixel at the bottom left corner of the screen
 * @param deltaU vector representing the change in the x direction
 * @param deltaV vector representing the change in the y direction
 * @param defocusDiskU vector representing the defocus disk in the x direction
 * @param defocusDiskV vector representing the defocus disk in the y direction
 * @param defocusAngle angle of the defocus disk
 * @param x x coordinate of the pixel
 * @param y y coordinate of the pixel
 * @param state curand state
 * @return cuda::std::tuple<Ray, Ray, Ray, Ray> generated rays
 */
__device__ auto get4Rays(
    const Vec3 &origin, const Vec3 &pixel00, const Vec3 &deltaU,
    const Vec3 &deltaV, const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
    const float defocusAngle, const uint16_t x, const uint16_t y,
    curandStatePhilox4_32_10_t &state) -> cuda::std::tuple<Ray, Ray, Ray, Ray>;