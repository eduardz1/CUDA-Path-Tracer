#include "cuda_path_tracer/camera.cuh"

__device__ auto defocusDiskSample(curandStatePhilox4_32_10_t &state,
                                  const Vec3 &center, const Vec3 &u,
                                  const Vec3 &v) -> Vec3 {
  const auto p = randomInUnitDiskRejectionSampling(state);
  return center + p.x * u + p.y * v;
}

__device__ auto defocusDiskSample(curandState_t &state, const Vec3 &center,
                                  const Vec3 &u, const Vec3 &v) -> Vec3 {
  const auto p = randomInUnitDiskRejectionSampling(state);
  return center + p.x * u + p.y * v;
}
__device__ auto
defocusDisk4Samples(curandStatePhilox4_32_10_t &state, const Vec3 &center,
                    const Vec3 &u,
                    const Vec3 &v) -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3> {
  const auto [p1, p2, p3, p4] = randomInUnitDisk(state);
  return {center + p1.x * u + p1.y * v, center + p2.x * u + p2.y * v,
          center + p3.x * u + p3.y * v, center + p4.x * u + p4.y * v};
}