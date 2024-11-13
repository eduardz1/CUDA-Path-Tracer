#include "cuda_path_tracer/error.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <iostream>

void cudaAssert(const cudaError_t code, const char *file, const int line) {
  if (code == cudaSuccess)
    return;

  std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " "
            << line << std::endl;

  exit(code);
}