#pragma once

#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#define DEGREE_TO_RADIAN(deg) (deg * M_PIf32 / 180.0f)

// TODO(eduard): This should be replaced with the official
// universal_host_pinned_vector implemented in
// https://github.com/NVIDIA/cccl/pull/2653 once a new release is available

using mr = thrust::universal_host_pinned_memory_resource;

template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, mr>;

template <typename T>
using universal_host_pinned_vector =
    thrust::host_vector<T, pinned_allocator<T>>;

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