#include "cuda_path_tracer/image.cuh"
#include <algorithm>
#include <climits>
#include <fstream>
#include <vector_functions.h>

namespace {
__device__ auto constexpr linToGamma(const float component) -> float {
  return component > 0 ? sqrtf(component) : 0.0F;
}
} // namespace

__host__ void saveImageAsPPM(const std::string &filename, const uint16_t width,
                             const uint16_t height,
                             const thrust::host_vector<uchar4> &image) {
  std::ofstream file(filename);

  file << "P3\n";
  file << width << " " << height << "\n";
  file << UCHAR_MAX << "\n";

  for (int i = 0; i < width * height; i++) {
    file << +image[i].x << " " << +image[i].y << " " << +image[i].z << "\n";
  }

  file.close();
}

__device__ auto convertColorTo8Bit(const Vec3 color) -> uchar4 {
  return make_uchar4(
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(linToGamma(color.x), 0.0F, 1.0F)),
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(linToGamma(color.y), 0.0F, 1.0F)),
      static_cast<unsigned char>(static_cast<float>(UCHAR_MAX) *
                                 std::clamp(linToGamma(color.z), 0.0F, 1.0F)),
      UCHAR_MAX);
}
