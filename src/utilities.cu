#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/utilities.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cmath>

StreamGuard::StreamGuard() { CUDA_ERROR_CHECK(cudaStreamCreate(&stream)) };

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

__device__ auto randomInUnitDiskRejectionSampling(curandState_t &state)
    -> Vec3 {
  while (true) {
    const auto p = Vec3{2.0F * curand_uniform(&state) - 1.0F,
                        2.0F * curand_uniform(&state) - 1.0F, 0};

    if (p.getLengthSquared() < 1.0F) {
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

    if (p.getLengthSquared() < 1.0F) {
      return p;
    }
    if (q.getLengthSquared() < 1.0F) {
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