#pragma once

#include "cuda_path_tracer/color.cuh"
#include "vec3.cuh"

/**
 * @brief Checker class, used to represent a checker texture
 */
class Checker {

public:
  __host__ __device__ Checker(double scale, Color even, Color odd)
      : scale(1.0 / scale), even(even), odd(odd) {}

  /**
   * @brief based on the point, return the color of the texture, which is either
   * colored accordingly to the even or odd color
   *
   * @param point point to get the texture color from
   * @return Color texture color
   */
  __device__ auto texture_value(const Vec3 &point) const -> Color {
    const auto x = static_cast<int>(std::floor(scale * point.x));
    const auto y = static_cast<int>(std::floor(scale * point.y));
    const auto z = static_cast<int>(std::floor(scale * point.z));

    const bool isEven = (x + y + z) % 2 == 0;

    return isEven ? even : odd;
  }

private:
  double scale;
  Color even;
  Color odd;
};
