#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/ray.cuh"
#include "cuda_path_tracer/vec3.cuh"

__host__ __device__ Ray::Ray() : origin(0), direction(0) {};
__host__ __device__ Ray::Ray(const Vec3 &origin, const Vec3 &direction)
    : origin(origin), direction(direction) {};

__host__ __device__ auto Ray::getOrigin() const -> Vec3 { return origin; }
__host__ __device__ auto Ray::getDirection() const -> Vec3 { return direction; }

__host__ __device__ auto Ray::at(const float t) const -> Vec3 {
  return origin + direction * t;
}

__device__ auto getRay(const Vec3 &origin, const Vec3 &pixel00,
                       const Vec3 &deltaU, const Vec3 &deltaV,
                       const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                       const float defocusAngle, const uint16_t x,
                       const uint16_t y, curandState_t &state) -> Ray {
  const auto a = curand_uniform(&state);
  const auto b = curand_uniform(&state);

  // We sample an area of "half pixel" around the pixel centers to achieve
  // anti-aliasing. This helps in reducing the jagged edges by averaging the
  // colors of multiple samples within each pixel, resulting in smoother
  // transitions and more realistic images.
  const auto offset = Vec3{a - 0.5F, b - 0.5F, 0};

  const auto sample = pixel00 + ((static_cast<float>(x) + offset.x) * deltaU) +
                      ((static_cast<float>(y) + offset.y) * deltaV);

  auto newOrigin = origin;

  if (defocusAngle > 0) {
    newOrigin = defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
  }

  const auto direction = sample - newOrigin;

  return {newOrigin, direction};
}

__device__ auto get2Rays(const Vec3 &origin, const Vec3 &pixel00,
                         const Vec3 &deltaU, const Vec3 &deltaV,
                         const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                         const float defocusAngle, const uint16_t x,
                         const uint16_t y, curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Ray, Ray> {
  const auto values = curand_uniform4(&state);

  const auto offsetA = Vec3{values.z - 0.5F, values.w - 0.5F, 0};
  const auto offsetB = Vec3{values.x - 0.5F, values.y - 0.5F, 0};

  const auto sampleA = pixel00 +
                       ((static_cast<float>(x) + offsetA.x) * deltaU) +
                       ((static_cast<float>(y) + offsetA.y) * deltaV);
  const auto sampleB = pixel00 +
                       ((static_cast<float>(x) + offsetB.x) * deltaU) +
                       ((static_cast<float>(y) + offsetB.y) * deltaV);

  auto newOriginA = origin;
  auto newOriginB = origin;

  if (defocusAngle > 0) {
    newOriginA = defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
    newOriginB = defocusDiskSample(state, origin, defocusDiskU, defocusDiskV);
  }

  const auto directionA = sampleA - newOriginA;
  const auto directionB = sampleB - newOriginB;

  return {{newOriginA, directionA}, {newOriginB, directionB}};
}

__device__ auto get4Rays(const Vec3 &origin, const Vec3 &pixel00,
                         const Vec3 &deltaU, const Vec3 &deltaV,
                         const Vec3 &defocusDiskU, const Vec3 &defocusDiskV,
                         const float defocusAngle, const uint16_t x,
                         const uint16_t y, curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Ray, Ray, Ray, Ray> {
  const auto values1 = curand_uniform4(&state);
  const auto values2 = curand_uniform4(&state);

  const auto offsetA = Vec3{values1.z - 0.5F, values1.w - 0.5F, 0};
  const auto offsetB = Vec3{values1.x - 0.5F, values1.y - 0.5F, 0};
  const auto offsetC = Vec3{values2.z - 0.5F, values2.w - 0.5F, 0};
  const auto offsetD = Vec3{values2.x - 0.5F, values2.y - 0.5F, 0};

  const auto sampleA = pixel00 +
                       ((static_cast<float>(x) + offsetA.x) * deltaU) +
                       ((static_cast<float>(y) + offsetA.y) * deltaV);
  const auto sampleB = pixel00 +
                       ((static_cast<float>(x) + offsetB.x) * deltaU) +
                       ((static_cast<float>(y) + offsetB.y) * deltaV);
  const auto sampleC = pixel00 +
                       ((static_cast<float>(x) + offsetC.x) * deltaU) +
                       ((static_cast<float>(y) + offsetC.y) * deltaV);
  const auto sampleD = pixel00 +
                       ((static_cast<float>(x) + offsetD.x) * deltaU) +
                       ((static_cast<float>(y) + offsetD.y) * deltaV);

  auto newOriginA = origin;
  auto newOriginB = origin;
  auto newOriginC = origin;
  auto newOriginD = origin;

  if (defocusAngle > 0) {
    const auto [s1, s2, s3, s4] =
        defocusDisk4Samples(state, origin, defocusDiskU, defocusDiskV);
    newOriginA = s1;
    newOriginB = s2;
    newOriginC = s3;
    newOriginD = s4;
  }

  const auto directionA = sampleA - newOriginA;
  const auto directionB = sampleB - newOriginB;

  return {{newOriginA, directionA},
          {newOriginB, directionB},
          {newOriginC, sampleC - newOriginC},
          {newOriginD, sampleD - newOriginD}};
}