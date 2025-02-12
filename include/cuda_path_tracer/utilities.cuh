#pragma once

#include "cuda_path_tracer/vec3.cuh"
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#define DEGREE_TO_RADIAN(deg)                                                  \
  (deg * 3.141592653589793238462643383279502884F / 180.0F)

#define WARP_SIZE 32

/**
 * @brief A guard for a CUDA stream that will automatically destroy the stream
 * when it goes out of scope. Uses the RAII idiom.
 */
class StreamGuard {
public:
  StreamGuard();

  ~StreamGuard();

  [[nodiscard]] auto get() const -> cudaStream_t;
  operator cudaStream_t() const;

  // We don't want it to be copyable
  StreamGuard(const StreamGuard &) = delete;
  auto operator=(const StreamGuard &) -> StreamGuard & = delete;

  // But it's fine to have it be moveable
  StreamGuard(StreamGuard &&other) noexcept;
  auto operator=(StreamGuard &&other) noexcept -> StreamGuard &;

private:
  cudaStream_t stream{};
};

template <class... Ts> struct overload : Ts... {
  using Ts::operator()...;
};

// Generate random points in a unit disk

/**
 * @brief Generate a random point in a unit disk through rejection sampling,
 * meaning that we will keep generating random points until we find one that is
 * within the unit disk.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit disk
 */
__device__ auto randomInUnitDiskRejectionSampling(curandState_t &state) -> Vec3;

/**
 * @brief Generate a random point in a unit disk through rejection sampling but
 * two choices at a time to reduce the number of iterations from pi/4 to pi/2 on
 * average. Furthermore, a single call of `curand_uniform4` is used to generate
 * four random numbers at once instead of two calls to `curand_uniform` like in
 * the `curandState_t` version.
 *
 * @param state The curand state
 * @return Vec3 The random point in the unit disk
 */
__device__ auto
randomInUnitDiskRejectionSampling(curandStatePhilox4_32_10_t &state) -> Vec3;

/**
 * @brief Generates four random points at a time in a unit disk by extrapolating
 * the density function of the disk.
 *
 * @param state The curand state
 * @return cuda::std::tuple<Vec3, Vec3, Vec3, Vec3> The four random points in
 * the unit disk
 */
__device__ auto randomInUnitDisk(curandStatePhilox4_32_10_t &state)
    -> cuda::std::tuple<Vec3, Vec3, Vec3, Vec3>;