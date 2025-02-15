#pragma once

#include "cuda_path_tracer/vec3.cuh"

struct TrigValues {
  float sin;
  float cos;

  __host__ __device__ constexpr TrigValues() : sin(0.0F), cos(1.0F) {}
  __host__ __device__ constexpr TrigValues(float sin, float cos)
      : sin(sin), cos(cos) {}
};

class Rotation {
protected:
  /**
   * @brief Rotates a point relative to the rotation of the rectangular cuboid
   *
   * @param point Point to rotate
   * @param inverse If true, the point is rotated in the opposite direction
   * @return Vec3 Rotated point
   */
  __host__ __device__ auto rotate(const Vec3 &point,
                                  const bool inverse) const -> Vec3;

  __host__ auto operator+=(const Rotation &r) -> Rotation &;

  __host__ Rotation(const Vec3 &angles);

  friend class RectangularCuboid;

private:
  Vec3 angles;

  struct TrigValues x, y, z;

  __host__ auto cacheTrigValues() -> void;
};
