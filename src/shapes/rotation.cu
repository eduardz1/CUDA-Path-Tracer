#include "cuda_path_tracer/shapes/rotation.cuh"
#include "cuda_path_tracer/utilities.cuh"

__host__ Rotation::Rotation(const Vec3 &angles) : angles(angles) {
  this->cacheTrigValues();
};

__host__ auto Rotation::cacheTrigValues() -> void {
  struct {
    float x, y, z;
  } angles_rad = {DEGREE_TO_RADIAN(angles.getX()),
                  DEGREE_TO_RADIAN(angles.getY()),
                  DEGREE_TO_RADIAN(angles.getZ())};

  x = {
      std::sin(angles_rad.x),
      std::cos(angles_rad.x),
  };
  y = {
      std::sin(angles_rad.y),
      std::cos(angles_rad.y),
  };
  z = {
      std::sin(angles_rad.z),
      std::cos(angles_rad.z),
  };
};

__host__ auto Rotation::operator+=(const Rotation &r) -> Rotation & {
  this->angles += r.angles;
  this->cacheTrigValues();
  return *this;
};

__device__ auto Rotation::rotatePoint(const Vec3 &point,
                                      const bool inverse) const -> Vec3 {
  struct {
    float cos, sin;
  } x{}, y{}, z{};
  x = {
      this->x.cos,
      this->x.sin,
  };
  y = {
      this->y.cos,
      this->y.sin,
  };
  z = {
      this->z.cos,
      this->z.sin,
  };

  if (inverse) {
    x.sin = -x.sin;
    y.sin = -y.sin;
    z.sin = -z.sin;
  }

  const auto x_rot = Vec3{
      point.getX(),
      x.cos * point.getY() - x.sin * point.getZ(),
      x.sin * point.getY() + x.cos * point.getZ(),
  };
  const auto y_rot = Vec3{
      y.cos * x_rot.getX() + y.sin * x_rot.getZ(),
      x_rot.getY(),
      -y.sin * x_rot.getX() + y.cos * x_rot.getZ(),
  };
  const auto z_rot = Vec3{
      z.cos * y_rot.getX() - z.sin * y_rot.getY(),
      z.sin * y_rot.getX() + z.cos * y_rot.getY(),
      y_rot.getZ(),
  };

  return z_rot;
};