#include "cuda_path_tracer/image.cuh"
#include <climits>
#include <fstream>
#include <vector_functions.h>

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
