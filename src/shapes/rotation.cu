#include "cuda_path_tracer/shapes/rotation.cuh"
#include "cuda_path_tracer/utilities.cuh"

__host__ Rotation::Rotation(const Vec3 &angles) : angles(angles) {
  this->cacheTrigValues();
};

__host__ auto Rotation::cacheTrigValues() -> void {
  struct {
#ifdef __NVCC__
#pragma nv_diag_suppress 2361
#endif
    float x, y, z;
  } angles_rad = {DEGREE_TO_RADIAN(angles.x), DEGREE_TO_RADIAN(angles.y),
                  DEGREE_TO_RADIAN(angles.z)};

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

__device__ auto Rotation::rotate(const Vec3 &point,
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
};