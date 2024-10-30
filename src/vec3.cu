#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ vec3::vec3() : x(0), y(0), z(0) {}
__host__ __device__ vec3::vec3(float value) : x(value), y(value), z(value) {}
__host__ __device__ vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) {}

__host__ __device__ auto vec3::getX() const -> float { return x; }
__host__ __device__ auto vec3::getY() const -> float { return y; }
__host__ __device__ auto vec3::getZ() const -> float { return z; }

__host__ __device__ auto vec3::operator+(const vec3 &other) const -> vec3 {
  return {x + other.x, y + other.y, z + other.z};
}
__host__ __device__ auto vec3::operator-(const vec3 &other) const -> vec3 {
  return {x - other.x, y - other.y, z - other.z};
}
__host__ __device__ auto vec3::operator*(const vec3 &other) const -> vec3 {
  return {x * other.x, y * other.y, z * other.z};
}
__host__ __device__ auto vec3::operator*(float t) const -> vec3 {
  return {x * t, y * t, z * t};
}
__host__ __device__ auto vec3::operator/(const vec3 &other) const -> vec3 {
  return {x / other.x, y / other.y, z / other.z};
}
__host__ __device__ auto vec3::operator==(const vec3 &other) const -> bool {
  return x == other.x && y == other.y && z == other.z;
}

__host__ __device__ auto vec3::dot(const vec3 &other) const -> float {
  return x * other.x + y * other.y + z * other.z;
}

__host__ auto operator<<(std::ostream &os, const vec3 &v) -> std::ostream & {
  os << "(" << v.getX() << ", " << v.getY() << ", " << v.getZ() << ")";
  return os;
}
