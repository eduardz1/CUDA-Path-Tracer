#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ Vec3::Vec3() : x(0), y(0), z(0) {}
__host__ __device__ Vec3::Vec3(const float value)
    : x(value), y(value), z(value) {}
__host__ __device__ Vec3::Vec3(const float x, const float y, const float z)
    : x(x), y(y), z(z) {}
__host__ __device__ Vec3::Vec3(const float4 &v) : x(v.x), y(v.y), z(v.z) {}

__host__ __device__ auto Vec3::operator-() const -> Vec3 {
  return {-x, -y, -z};
}
__host__ __device__ auto Vec3::operator==(const Vec3 &other) const -> bool {
  return x == other.x && y == other.y && z == other.z;
}
__host__ __device__ auto Vec3::operator+=(const Vec3 &other) -> Vec3 & {
  x += other.x;
  y += other.y;
  z += other.z;

  return *this;
}
__device__ Vec3::operator float4() const { return make_float4(x, y, z, 1.0f); }

__host__ __device__ auto Vec3::getLengthSquared() const -> float {
  return x * x + y * y + z * z;
}

__host__ __device__ auto Vec3::getLength() const -> float {
  return sqrt(getLengthSquared());
}

__host__ auto operator<<(std::ostream &os, const Vec3 &v) -> std::ostream & {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}
__host__ __device__ auto operator+(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}
__host__ __device__ auto operator-(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}
__host__ __device__ auto operator*(const Vec3 &v, float t) -> Vec3 {
  return {t * v.x, t * v.y, t * v.z};
}
__host__ __device__ auto operator*(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}
__host__ __device__ auto operator/(const Vec3 &v, float t) -> Vec3 {
  return {v.x / t, v.y / t, v.z / t};
}

__host__ __device__ auto makeUnitVector(const Vec3 &v) -> Vec3 {
  return v / v.getLength();
}

__host__ __device__ auto dot(const Vec3 &v1, const Vec3 &v2) -> float {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ auto cross(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
          v1.x * v2.y - v1.y * v2.x};
}