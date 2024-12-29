#include "cuda_path_tracer/vec3.cuh"

#pragma once

class Rotation {
protected:
  /**
   * @brief Rotates a point relative to the rotation of the rectangular cuboid
   *
   * @param point Point to rotate
   * @param inverse If true, the point is rotated in the opposite direction
   * @return Vec3 Rotated point
   */
  __device__ auto rotatePoint(const Vec3 &point,
                              const bool inverse) const -> Vec3;

  __host__ auto operator+=(const Rotation &r) -> Rotation &;

  __host__ Rotation() = default;
  __host__ Rotation(const Vec3 &angles);

  friend class RectangularCuboid;

private:
  Vec3 angles{};

  struct {
    float sin{}, cos = 1;
  } x, y, z;
  __host__ auto cacheTrigValues() -> void;
};