#include "cuda_path_tracer/error.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess)
    return;

  fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
          line);
  exit(code);
}