#pragma once
#include "vec3.cuh"

class Checker {

public:
  __host__ __device__ Checker(double scale, Vec3 even, Vec3 odd)
      : scale(1.0 / scale), even(even), odd(odd) {}

  __device__ Vec3 texture_value(Vec3 &point) {
    auto x = int(std::floor(scale * point.x));
    auto y = int(std::floor(scale * point.y));
    auto z = int(std::floor(scale * point.z));

    bool isEven = (x + y + z) % 2 == 0;
    return isEven ? even : odd;
  }

private:
  double scale;
  Vec3 even;
  Vec3 odd;
};
