#pragma once

#include <driver_types.h>

#define CUDA_ERROR_CHECK(ans)                                                  \
  { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line);