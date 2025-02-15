#pragma once

#include "vec3.cuh"

class Checker {

public:
  __host__ __device__ Checker(double scale, Vec3 even, Vec3 odd)
      : scale(1.0 / scale), even(even), odd(odd) {}

  __device__ auto texture_value(const Vec3 &point) const -> Vec3 {
    auto x = static_cast<int>(std::floor(scale * point.x));
    auto y = static_cast<int>(std::floor(scale * point.y));
    auto z = static_cast<int>(std::floor(scale * point.z));

    bool isEven = (x + y + z) % 2 == 0;
    return isEven ? even : odd;
  }

private:
  double scale;
  Vec3 even;
  Vec3 odd;
};
