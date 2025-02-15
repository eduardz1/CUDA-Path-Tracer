#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/utilities.cuh"

StreamGuard::StreamGuard(){CUDA_ERROR_CHECK(cudaStreamCreate(&stream))};

StreamGuard::~StreamGuard() {
  if (stream != nullptr) {
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream))
  }
};

[[nodiscard]] auto StreamGuard::get() const -> cudaStream_t { return stream; }
StreamGuard::operator cudaStream_t() const { return stream; }

auto StreamGuard::operator=(StreamGuard &&other) noexcept -> StreamGuard & {
  if (this != &other) {
    if (stream != nullptr) {
      CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
    }
    stream = other.stream;
    other.stream = nullptr;
  }
  return *this;
}
StreamGuard::StreamGuard(StreamGuard &&other) noexcept : stream(other.stream) {
  other.stream = nullptr;
}
