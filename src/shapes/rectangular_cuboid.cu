#include "cuda_path_tracer/shapes/rectangular_cuboid.cuh"
#include "cuda_path_tracer/shapes/rotation.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cuda/std/array>

__host__ RectangularCuboid::RectangularCuboid(const Vec3 &a, const Vec3 &b,
                                              const Material &material)
    : material(material) {
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
}

__host__ RectangularCuboid::RectangularCuboid(const Faces &transformed_faces,
                                              const Material &material)
    : faces(transformed_faces), material(material) {}

__host__ auto RectangularCuboid::rotate(const Vec3 &angles) const
    -> RectangularCuboid {
  // Create a rotation instance.
  const auto rot = Rotation(angles);

  // Compute center of cuboid using the bottom and top faces.
  const Vec3 bottom_center =
      faces.bottom.origin + 0.5F * (faces.bottom.u + faces.bottom.v);
  const Vec3 top_center = faces.top.origin + 0.5F * (faces.top.u + faces.top.v);
  const Vec3 center = 0.5F * (bottom_center + top_center);

  // For each face, rotate its origin about the cuboid center and rotate its
  // edge vectors.
  auto rotate_face = [rot, center,
                      this](const Parallelogram &face) -> Parallelogram {
    const Vec3 new_origin = center + rot.rotate(face.origin - center, false);
    const Vec3 new_u = rot.rotate(face.u, false);
    const Vec3 new_v = rot.rotate(face.v, false);
    return {new_origin, new_u, new_v, material};
  };

  return {{.front = rotate_face(faces.left),
           .back = rotate_face(faces.bottom),
           .left = rotate_face(faces.front),
           .right = rotate_face(faces.right),
           .top = rotate_face(faces.back),
           .bottom = rotate_face(faces.top)},
          material};
}

__host__ auto RectangularCuboid::translate(const Vec3 &offset) const
    -> RectangularCuboid {
  return {{
              .front = {faces.left.origin + offset, faces.left.u, faces.left.v,
                        material},
              .back = {faces.bottom.origin + offset, faces.bottom.u,
                       faces.bottom.v, material},
              .left = {faces.front.origin + offset, faces.front.u,
                       faces.front.v, material},
              .right = {faces.right.origin + offset, faces.right.u,
                        faces.right.v, material},
              .top = {faces.back.origin + offset, faces.back.u, faces.back.v,
                      material},
              .bottom = {faces.top.origin + offset, faces.top.u, faces.top.v,
                         material},
          },
          material};
}

__device__ auto RectangularCuboid::hit(const Ray &r, const float hit_t_min,
                                       const float hit_t_max, HitInfo &hi) const
    -> bool {
  HitInfo temp_hi;
  bool hit_any = false;
  float closest_t = hit_t_max;

  cuda::std::array faces_arr{&faces.left,  &faces.bottom, &faces.front,
                             &faces.right, &faces.back,   &faces.top};

  for (const auto &i : faces_arr) {
    if (i->hit(r, hit_t_min, closest_t, temp_hi)) {
      hit_any = true;
      closest_t = temp_hi.time;
      hi = temp_hi;
    }
  }

  return hit_any;
}