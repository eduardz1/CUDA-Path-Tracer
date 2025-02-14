#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/array>

__host__ RectangularCuboid::RectangularCuboid(const Vec3 &a, const Vec3 &b, const Material &material)
    : a(a), b(b), material(material) {
  const auto min =
      Vec3(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
  const auto max =
      Vec3(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));

  const auto dx = Vec3(max.x - min.x, 0, 0);
  const auto dy = Vec3(0, max.y - min.y, 0);
  const auto dz = Vec3(0, 0, max.z - min.z);

  faces.left = Parallelogram(min, dz, dy, material);
  faces.bottom = Parallelogram(min, dx, dz, material);
  faces.front = Parallelogram({min.x, min.y, max.z}, dx, dy, material);
  faces.right = Parallelogram({max.x, min.y, max.z}, -dz, dy, material);
  faces.back = Parallelogram({max.x, min.y, min.z}, -dx, dy, material);
  faces.top = Parallelogram({min.x, max.y, max.z}, dx, -dz, material);
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
  const auto origin = rotation.rotate(r.getOrigin(), true);
  const auto direction = rotation.rotate(r.getDirection(), true);
  const Ray rotated_ray = {origin - this->translation, direction};

  HitInfo temp_hi;
  bool hit_any = false;
  float closest_t = hit_t_max;

  cuda::std::array faces_arr{&faces.left,  &faces.bottom, &faces.front,
                             &faces.right, &faces.back,   &faces.top};

  for (const auto &i : faces_arr) {
    if (i->hit(rotated_ray, hit_t_min, closest_t, temp_hi)) {
      hit_any = true;
      closest_t = temp_hi.time;
      hi = temp_hi;
    }
  }

  if (!hit_any) {
    return false;
  }

  hi.point = rotation.rotate(hi.point, false) + this->translation;
  hi.normal = rotation.rotate(hi.normal, false);
  hi.material = material;

  return true;
}
