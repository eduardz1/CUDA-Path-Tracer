#pragma once

#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#define DEGREE_TO_RADIAN(deg) (deg * M_PIf32 / 180.0f)

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