#include "cuda_path_tracer/world.cuh"
#include <climits>
#include <cmath>
#include <vector_functions.h>

__device__ __host__ auto groundColor(const Vec3 &origin,
                                     const Vec3 &direction) -> uchar4 {
  // TODO: Implement ground color
  return make_uchar4(0, 0, 0, UCHAR_MAX);
}

__device__ __host__ auto skyColor(const Vec3 &direction) -> uchar4 {
  auto sky_color = SKY_COLOR;

  auto scaling_factor = std::pow(1.0f - direction.getY(), 2.0f);

  sky_color.x = static_cast<unsigned char>(static_cast<float>(sky_color.x) *
                                           scaling_factor);
  sky_color.y = static_cast<unsigned char>(static_cast<float>(sky_color.y) *
                                           scaling_factor);
  sky_color.z = static_cast<unsigned char>(static_cast<float>(sky_color.z) *
                                           scaling_factor);

  return sky_color;
}