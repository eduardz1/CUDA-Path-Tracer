#pragma once

#include "cuda_path_tracer/hit_info.cuh"
#include "cuda_path_tracer/shapes/parallelogram.cuh"
#include "cuda_path_tracer/shapes/rotation.cuh"

class RectangularCuboid {
public:
  __host__ RectangularCuboid(const Vec3 &a, const Vec3 &b, const Material &material);

  /**
   * @brief Saves the normal vector of the rectangular cuboid at the point of
   * intersection in the HitInfo class and returns true if the ray intersects
   * the rectangular cuboid, false otherwise. in our implementation, the normal
   * vector is of unit length and is calculated by subtracting the center from
   * the point of intersection.
   *
   * @param r Ray that intersects the rectangular cuboid
   * @param hit_t_min minimum time value for the intersection
   * @param hit_t_max maximum time value for the intersection
   * @param hi HitInfo class that contains the point of intersection and the
   * normal vector at a given time
   * @return bool true if the ray intersects the rectangular cuboid, false
   * otherwise
   */
  __device__ auto hit(const Ray &r, const float hit_t_min,
                      const float hit_t_max, HitInfo &hi) const -> bool;

  __host__ auto rotate(const Vec3 &angles) -> RectangularCuboid &;
  __host__ auto translate(const Vec3 &translation) -> RectangularCuboid &;

private:
  Vec3 a, b, translation;

  // Always performs the rotation first, then the translation

  Rotation rotation;
  Material material;

  struct {
    Parallelogram front, back, left, right, top, bottom;
  } faces;
};