#pragma once

#include "cuda_path_tracer/ray.cuh"

// minimum distance before considering a hit, to avoid self-intersection
#define RAY_T_MIN 0.001f

// maximum distance before considering a hit, to avoid infinite loops
#define RAY_T_MAX 100000.0f

/**
 * @brief Stores information about a ray hit on a Shape, including the time of
 * intersection, the point of intersection, and the normal vector at the point
 * of intersection
 */
class HitInfo {
public:
  __device__ HitInfo();
  __device__ HitInfo(const Vec3 &point, const Vec3 &normal, const float time);

  __device__ auto getPoint() const -> Vec3;
  __device__ auto getNormal() const -> Vec3;
  __device__ auto getTime() const -> float;
  __device__ auto getFront() const -> bool;

  __device__ void setPoint(const Vec3 &point);

  /**
   * @brief Set the Normal object, when a ray hits a Shape from outside, the
   * normal vector points against the ray, when the ray hits the Shape from
   * inside, the normal vector points in the same direction as the ray. This
   * function sets the normal vector so that it always points against the ray.
   *
   * @param r Ray object that hit the Shape
   * @param outward_normal Normal vector at the point of intersection
   */
  __device__ void setNormal(const Ray &r, const Vec3 &outward_normal);
  __device__ void setTime(const float time);

private:
  Vec3 point, normal;
  float time;

  /**
   * @brief Stores whether the hit is on the front or back of the Shape
   */
  bool front;
};