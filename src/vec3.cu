#include "cuda_path_tracer/vec3.cuh"
#include <random>

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
__device__ Vec3::operator float4() const { return make_float4(x, y, z, 1.0F); }

__host__ __device__ auto Vec3::getLengthSquared() const -> float {
  return x * x + y * y + z * z;
}

__host__ __device__ auto Vec3::getLength() const -> float {
  return sqrtf(getLengthSquared());
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
__host__ __device__ auto operator*(const Vec3 &v, const float t) -> Vec3 {
  return {t * v.x, t * v.y, t * v.z};
}
__host__ __device__ auto operator*(const Vec3 &v1, const Vec3 &v2) -> Vec3 {
  return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}
__host__ __device__ auto operator/(const Vec3 &v, float t) -> Vec3 {
  return {v.x / t, v.y / t, v.z / t};
}

__device__ auto randomVector(curandState &state) -> Vec3 {
  return Vec3{curand_uniform(&state), curand_uniform(&state),
              curand_uniform(&state)};
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

__device__ auto vectorOnHemisphere(const Vec3 &v, curandState &state) -> Vec3 {
  Vec3 randomUnit = makeUnitVector(randomVector(state));

  if (dot(randomUnit, v) > 0.0) {
    return randomUnit;
  }
  return -randomUnit;
}

__device__ auto roundScatterDirection(const Vec3 &direction,
                                      const Vec3 &normal) -> Vec3 {
  auto s = 1e-8;
  if (fabs(direction.x < s) && fabs(direction.y < s) &&
      fabs(direction.z < s)) {
    return normal;
  }
  return direction;
}

__device__ auto reflect(const Vec3 &v, const Vec3 &n) -> Vec3 {
  return v - 2 * dot(v, n) * n;
}

__device__ auto refract(const Vec3 &v, const Vec3 &n,
                        double eta_component) -> Vec3 {
  auto cos_theta = fmin(dot(-v, n), 1.0);
  Vec3 R_perp = eta_component * (v + cos_theta * n);
  Vec3 R_par = -sqrt(fabs(1.0f - R_perp.getLengthSquared())) * n;
  return R_perp + R_par;
}

__device__ auto vectorOnHemisphere(const Vec3 &v, curandState &state) -> Vec3 {
  Vec3 randomUnit = makeUnitVector(randomVector(state));

  if (dot(randomUnit, v) > 0.0) {
    return randomUnit;
  }
  return -randomUnit;
}

__device__ auto roundScatterDirection(const Vec3 &direction,
                                      const Vec3 &normal) -> Vec3 {
  auto s = 1e-8;
  if (fabs(direction.x < s) && fabs(direction.y < s) &&
      fabs(direction.z < s)) {
    return normal;
  }
  return direction;
}

__device__ auto reflect(const Vec3 &v, const Vec3 &n) -> Vec3 {
  return v - 2 * dot(v, n) * n;
}

__device__ auto refract(const Vec3 &v, const Vec3 &n,
                        double eta_component) -> Vec3 {
  auto cos_theta = fmin(dot(-v, n), 1.0);
  Vec3 R_perp = eta_component * (v + cos_theta * n);
  Vec3 R_par = -sqrt(fabs(1.0f - R_perp.getLengthSquared())) * n;
  return R_perp + R_par;
}