#pragma once

#include "cuda_path_tracer/color.cuh"
#include "vec3.cuh"

class Checker {

public:
  __host__ __device__ Checker(double scale, Color even, Color odd)
      : scale(1.0 / scale), even(even), odd(odd) {}

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
