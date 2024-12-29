#pragma once

#include "cuda_path_tracer/hit_info.cuh"

class Parallelogram {
public:
  __host__ Parallelogram();
  __host__ Parallelogram(const Vec3 &origin, const Vec3 &u, const Vec3 &v);

  /**
   * @brief Saves the normal vector of the parallelogram at the point of
   * intersection in the HitInfo class and returns true if the ray intersects
   * the parallelogram, false otherwise. in our implementation, the normal
   * vector is of unit length and is calculated by subtracting the center from
   * the point of intersection.
   *
   * @param r Ray that intersects the parallelogram
   * @param hit_t_min minimum time value for the intersection
   * @param hit_t_max maximum time value for the intersection
   * @param hi HitInfo class that contains the point of intersection and the
   * normal vector at a given time
   * @return bool true if the ray intersects the parallelogram, false otherwise
   */
  __device__ auto hit(const Ray &r, const float hit_t_min,
                      const float hit_t_max, HitInfo &hi) const -> bool;

private:
  Vec3 origin, u, v, w, normal;
  float area;
};