#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/array>

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
  this->translation += translation;
  return *this;
};

__device__ auto RectangularCuboid::hit(const Ray &r, const float hit_t_min,
                                       const float hit_t_max,
                                       HitInfo &hi) const -> bool {
  // First translate, then rotate (inverse transformations are applied in
  // reverse)
  const auto origin = rotation.rotate(r.getOrigin() - translation, true);
  const auto direction = rotation.rotate(r.getDirection(), true);
  const Ray transformed_ray{origin, direction};

  HitInfo temp_hi;
  bool hit_any = false;
  float closest_t = hit_t_max;

  cuda::std::array faces_arr{&faces.left,  &faces.bottom, &faces.front,
                             &faces.right, &faces.back,   &faces.top};

  for (const auto &i : faces_arr) {
    if (i->hit(transformed_ray, hit_t_min, closest_t, temp_hi)) {
      hit_any = true;
      closest_t = temp_hi.time;
      hi = temp_hi;
    }
  }

  if (!hit_any) {
    return false;
  }

  // Apply transformations in forward order: first rotation, then translation
  hi.point = rotation.rotate(hi.point, false) + translation;
  hi.normal = makeUnitVector(rotation.rotate(hi.normal, false));
  hi.time = closest_t;

  return true;
}
