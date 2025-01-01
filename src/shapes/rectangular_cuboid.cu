#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <array>

__host__ RectangularCuboid::RectangularCuboid(const Vec3 &a, const Vec3 &b)
    : a(a), b(b) {
  const auto min =
      Vec3(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
  const auto max =
      Vec3(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));

  const auto dx = Vec3(max.x - min.x, 0, 0);
  const auto dy = Vec3(0, max.y - min.y, 0);
  const auto dz = Vec3(0, 0, max.z - min.z);

  faces.left = Parallelogram(min, dz, dy);
  faces.bottom = Parallelogram(min, dx, dz);
  faces.front = Parallelogram({min.x, min.y, max.z}, dx, dy);
  faces.right = Parallelogram({max.x, min.y, max.z}, -dz, dy);
  faces.back = Parallelogram({max.x, min.y, min.z}, -dx, dy);
  faces.top = Parallelogram({min.x, max.y, max.z}, dx, -dz);
};

__host__ auto
RectangularCuboid::rotate(const Vec3 &angles) -> RectangularCuboid & {
  this->rotation += Rotation(angles);
  return *this;
};
__host__ auto
RectangularCuboid::translate(const Vec3 &translation) -> RectangularCuboid & {
  this->translation += {-translation.x, translation.y, translation.z};
  return *this;
};

__device__ auto RectangularCuboid::hit(const Ray &r, const float hit_t_min,
                                       const float hit_t_max,
                                       HitInfo &hi) const -> bool {
  const auto origin = rotation.rotatePoint(r.getOrigin(), true);
  const auto direction = rotation.rotatePoint(r.getDirection(), true);
  const Ray rotated_ray = {origin - this->translation, direction};

  HitInfo temp_hi;
  bool hit_any = false;
  float closest_t = hit_t_max;

  std::array<const Parallelogram *, 6> faces_arr{// NOLINT
                                                 &faces.left,  &faces.bottom,
                                                 &faces.front, &faces.right,
                                                 &faces.back,  &faces.top};

  for (auto &i : faces_arr) {
    if (i->hit(rotated_ray, hit_t_min, closest_t, temp_hi)) {
      hit_any = true;
      closest_t = temp_hi.getTime();
      hi = temp_hi;
    }
  }

  if (!hit_any) {
    return false;
  }

  hi.setPoint(rotation.rotatePoint(hi.getPoint(), false) + this->translation);
  hi.setNormal(rotation.rotatePoint(hi.getNormal(), false));

  return true;
}
