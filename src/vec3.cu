#include "cuda_path_tracer/vec3.cuh"
#include <cmath>

constexpr auto epsilon = 1e-6F;

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
__host__ __device__ auto Vec3::operator*=(const Vec3 &other) -> Vec3 & {
  x *= other.x;
  y *= other.y;
  z *= other.z;

  return *this;
}
__host__ __device__ auto Vec3::operator/=(const float t) -> Vec3 & {
  const auto k = 1.0F / t;

  x *= k;
  y *= k;
  z *= k;

  return *this;
}
__device__ Vec3::operator float4() const { return make_float4(x, y, z, 1.0F); }

__host__ __device__ auto Vec3::getLengthSquared() const -> float {
  return x * x + y * y + z * z;
}

__host__ __device__ auto Vec3::getLength() const -> float {
  return sqrtf(getLengthSquared());
}

__host__ __device__ auto Vec3::nearZero() const -> bool {
  return std::fabs(x) < epsilon && std::fabs(y) < epsilon &&
         std::fabs(z) < epsilon;
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

__device__ auto randomVector(curandState_t &state) -> Vec3 {
  return Vec3{curand_uniform(&state), curand_uniform(&state),
              curand_uniform(&state)};
}
__device__ auto randomVector(curandStatePhilox4_32_10_t &state) -> Vec3 {
  const auto values = curand_uniform4(&state);
  return Vec3{values.x, values.y, values.z};
}

__device__ auto randomVector(curandState_t &state, const float min,
                             const float max) -> Vec3 {
  return Vec3{curand_uniform(&state) * (max - min) + min,
              curand_uniform(&state) * (max - min) + min,
              curand_uniform(&state) * (max - min) + min};
}

__device__ auto randomVector(curandStatePhilox4_32_10_t &state, const float min,
                             const float max) -> Vec3 {
  const auto values = curand_uniform4(&state);
  return Vec3{values.x * (max - min) + min, values.y * (max - min) + min,
              values.z * (max - min) + min};
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

template <typename State>
__device__ auto vectorOnHemisphere(const Vec3 &v, State &state) -> Vec3 {
  Vec3 randomUnit = makeUnitVector(randomVector(state));

  return dot(randomUnit, v) > 0.0F ? randomUnit : -randomUnit;
}

template __device__ auto
vectorOnHemisphere<curandState_t>(const Vec3 &, curandState_t &) -> Vec3;
template __device__ auto vectorOnHemisphere<curandStatePhilox4_32_10_t>(
    const Vec3 &, curandStatePhilox4_32_10_t &) -> Vec3;

__device__ auto roundScatterDirection(const Vec3 &direction,
                                      const Vec3 &normal) -> Vec3 {
  const auto s = 1e-8F;
  if (std::fabs(direction.x) < s && std::fabs(direction.y) < s &&
      std::fabs(direction.z) < s) {
    return normal;
  }
  return direction;
}

__device__ auto
randomInUnitDiskRejectionSampling(curandState_t &state) -> Vec3 {
  while (true) {
    const auto p = Vec3{2.0F * curand_uniform(&state) - 1.0F,
                        2.0F * curand_uniform(&state) - 1.0F, 0};

    const auto sqrd = p.getLengthSquared();

    if (epsilon < sqrd && sqrd <= 1.0F) {
      return p;
    }
  }
}

__device__ auto
randomInUnitDiskRejectionSampling(curandStatePhilox4_32_10_t &state) -> Vec3 {
  while (true) {
    const auto values = curand_uniform4(&state);

    const auto p = Vec3{2.0F * values.w - 1.0F, 2.0F * values.x - 1.0F, 0};
    const auto q = Vec3{2.0F * values.y - 1.0F, 2.0F * values.z - 1.0F, 0};

    const auto p_sqrd = p.getLengthSquared();
    const auto q_sqrd = q.getLengthSquared();

    if (epsilon < p_sqrd && p_sqrd <= 1.0F) {
      return p;
    }
    if (epsilon < q_sqrd && q_sqrd <= 1.0F) {
      return q;
    }
  }
}

__device__ auto randomInUnitDisk(curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3> {
  const float4 radius = curand_uniform4(&state);
  const float4 angle = curand_uniform4(&state) * 2.0F * M_PIf32;

  return cuda::std::tuple{
      Vec3{sqrtf(radius.x) * cosf(angle.x), sqrtf(radius.x) * sinf(angle.x), 0},
      Vec3{sqrtf(radius.y) * cosf(angle.y), sqrtf(radius.y) * sinf(angle.y), 0},
      Vec3{sqrtf(radius.z) * cosf(angle.z), sqrtf(radius.z) * sinf(angle.z), 0},
      Vec3{sqrtf(radius.w) * cosf(angle.w), sqrtf(radius.w) * sinf(angle.w), 0},
  };
}