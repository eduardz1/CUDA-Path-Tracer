#include "cuda_path_tracer/shapes/rotation.cuh"
#include "cuda_path_tracer/utilities.cuh"

__host__ Rotation::Rotation(const Vec3 &angles) : angles(angles) {
  this->cacheTrigValues();
};

__host__ auto Rotation::cacheTrigValues() -> void {
  const Vec3 angles_rad = {
      DEGREE_TO_RADIAN(angles.x),
      DEGREE_TO_RADIAN(angles.y),
      DEGREE_TO_RADIAN(angles.z),
  };

  this->x = {
      std::sin(angles_rad.x),
      std::cos(angles_rad.x),
  };
  this->y = {
      std::sin(angles_rad.y),
      std::cos(angles_rad.y),
  };
  this->z = {
      std::sin(angles_rad.z),
      std::cos(angles_rad.z),
  };
};

__host__ auto Rotation::operator+=(const Rotation &r) -> Rotation & {
  this->angles += r.angles;
  this->cacheTrigValues();
  return *this;
};

__device__ auto Rotation::rotate(const Vec3 &point,
                                 const bool inverse) const -> Vec3 {
  if (!inverse) { // Forward rotation (x -> y -> z)
    const auto x_rot = Vec3{
        point.x,
        x.cos * point.y - x.sin * point.z,
        x.sin * point.y + x.cos * point.z,
    };
    const auto y_rot = Vec3{
        y.cos * x_rot.x + y.sin * x_rot.z,
        x_rot.y,
        -y.sin * x_rot.x + y.cos * x_rot.z,
    };
    const auto z_rot = Vec3{
        z.cos * y_rot.x - z.sin * y_rot.y,
        z.sin * y_rot.x + z.cos * y_rot.y,
        y_rot.z,
    };
    return z_rot;
  }

  // Inverse rotation (z -> y -> x)
  // Use negative angles by negating both sin and keeping cos
  const auto z_inv = Vec3{
      z.cos * point.x + z.sin * point.y,
      -z.sin * point.x + z.cos * point.y,
      point.z,
  };
  const auto y_inv = Vec3{
      y.cos * z_inv.x - y.sin * z_inv.z,
      z_inv.y,
      y.sin * z_inv.x + y.cos * z_inv.z,
  };
  const auto x_inv = Vec3{
      y_inv.x,
      x.cos * y_inv.y + x.sin * y_inv.z,
      -x.sin * y_inv.y + x.cos * y_inv.z,
  };
  return x_inv;
};