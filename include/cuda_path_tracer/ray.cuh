#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/tuple>

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

__device__ auto getRay(const Vec3 &origin, const Vec3 &pixel00,
                       const Vec3 &deltaU, const Vec3 &deltaV,
                       const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                       const float defocusAngle, const uint16_t x,
                       const uint16_t y, curandState_t &state) -> Ray;

__device__ auto get2Rays(const Vec3 &origin, const Vec3 &pixel00,
                         const Vec3 &deltaU, const Vec3 &deltaV,
                         const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                         const float defocusAngle, const uint16_t x,
                         const uint16_t y, curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Ray, Ray>;

__device__ auto get4Rays(const Vec3 &origin, const Vec3 &pixel00,
                         const Vec3 &deltaU, const Vec3 &deltaV,
                         const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                         const float defocusAngle, const uint16_t x,
                         const uint16_t y, curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Ray, Ray, Ray, Ray>;