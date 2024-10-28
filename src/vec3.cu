#include "cuda_path_tracer/vec3.cuh"

vec3::vec3() : x(0), y(0), z(0) {}
vec3::vec3(float value) : x(value), y(value), z(value) {}
vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) {}

auto vec3::getX() const -> float { return x; }
auto vec3::getY() const -> float { return y; }
auto vec3::getZ() const -> float { return z; }

auto vec3::operator+(const vec3 &other) const -> vec3 {
  return {x + other.x, y + other.y, z + other.z};
}
auto vec3::operator-(const vec3 &other) const -> vec3 {
  return {x - other.x, y - other.y, z - other.z};
}
auto vec3::operator*(const vec3 &other) const -> vec3 {
  return {x * other.x, y * other.y, z * other.z};
}
auto vec3::operator*(float t) const -> vec3 { return {x * t, y * t, z * t}; }
auto vec3::operator/(const vec3 &other) const -> vec3 {
  return {x / other.x, y / other.y, z / other.z};
}
auto vec3::operator==(const vec3 &other) const -> bool {
  return x == other.x && y == other.y && z == other.z;
}

auto vec3::dot(const vec3 &other) const -> float {
  return x * other.x + y * other.y + z * other.z;
}

auto operator<<(std::ostream &os, const vec3 &v) -> std::ostream & {
  os << "(" << v.getX() << ", " << v.getY() << ", " << v.getZ() << ")";
  return os;
}
