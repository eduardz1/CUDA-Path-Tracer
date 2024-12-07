#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ Vec3::Vec3() : x(0), y(0), z(0) {}
__host__ __device__ Vec3::Vec3(const float value)
    : x(value), y(value), z(value) {}
__host__ __device__ Vec3::Vec3(const float x, const float y, const float z)
    : x(x), y(y), z(z) {}

__host__ __device__ auto Vec3::getX() const -> float { return x; }
__host__ __device__ auto Vec3::getY() const -> float { return y; }
__host__ __device__ auto Vec3::getZ() const -> float { return z; }

__host__ __device__ auto Vec3::operator+(const Vec3 &other) const -> Vec3 {
  return {x + other.x, y + other.y, z + other.z};
}
__host__ __device__ auto Vec3::operator-(const Vec3 &other) const -> Vec3 {
  return {x - other.x, y - other.y, z - other.z};
}
__host__ __device__ auto Vec3::operator/(const Vec3 &other) const -> Vec3 {
  return {x / other.x, y / other.y, z / other.z};
}
__host__ __device__ auto Vec3::operator==(const Vec3 &other) const -> bool {
  return x == other.x && y == other.y && z == other.z;
}

__host__ __device__ auto Vec3::dot(const Vec3 &other) const -> float {
  return x * other.x + y * other.y + z * other.z;
}

__host__ auto operator<<(std::ostream &os, const Vec3 &v) -> std::ostream & {
  os << "(" << v.getX() << ", " << v.getY() << ", " << v.getZ() << ")";
  return os;
}
__host__ __device__ auto operator*(const Vec3 &v, float t) -> Vec3 {
  return {t * v.getX(), t * v.getY(), t * v.getZ()};
}
__host__ __device__ auto operator*(float t, const Vec3 &v) -> Vec3 {
  return {t * v.getX(), t * v.getY(), t * v.getZ()};
}
__host__ __device__ auto operator*(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.getX() * v2.getX(), v1.getY() * v2.getY(), v1.getZ() * v2.getZ()};
}