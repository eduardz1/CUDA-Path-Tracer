#pragma once

#include "cuda_path_tracer/hit_info.cuh"

class Sphere {
public:

  __host__ Sphere(const Vec3 &center, float radius, const Material &material);

  /**
   * @brief Saves the normal vector of the sphere at the point of intersection
   * in the HitInfo class and returns true if the ray intersects the sphere,
   * false otherwise. in our implementation, the normal vector is of unit length
   * and is calculated by subtracting the center from the point of intersection.
   *
   * @param r Ray that intersects the sphere
   * @param hit_t_min minimum time value for the intersection
   * @param hit_t_max maximum time value for the intersection
   * @param hi HitInfo class that contains the point of intersection and the
   * normal vector at a given time
   * @return bool true if the ray intersects the sphere, false otherwise
   */
  __device__ auto hit(const Ray &r, const float hit_t_min,
                      const float hit_t_max, HitInfo &hi) const -> bool;
  __device__ auto getCenter() const -> Vec3;
  __device__ auto getMaterial() const -> Material;

private:
  Vec3 center;
  float radius;
  Material material;
};