#pragma once

#include <driver_types.h>

#define CUDA_ERROR_CHECK(ans)                                                  \
  { cudaAssert((ans), __FILE__, __LINE__); }

/**
 * @brief Check if a CUDA error occurred and print the error message.
 *
 * @param code cuda error code
 * @param file file where the error occurred
 * @param line line where the error occurred
 */
void cudaAssert(cudaError_t code, const char *file, int line);