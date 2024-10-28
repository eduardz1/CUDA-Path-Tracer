#include "cuda_path_tracer/error.cuh"
#include <catch2/catch_test_macros.hpp>
#include <driver_types.h>
#include <iostream>

// Mock exit function
void exit(int code) { throw code; }

TEST_CASE("cudaAssert function", "[cudaAssert]") {
  // Redirect stderr to a string stream
  std::stringstream buffer;
  std::streambuf *old = std::cerr.rdbuf(buffer.rdbuf());

  // Test case where cudaSuccess is returned
  REQUIRE_NOTHROW(cudaAssert(cudaSuccess, __FILE__, __LINE__));

  // Test case where an error code is returned
  cudaError_t errorCode = cudaErrorMemoryAllocation;
  try {
    cudaAssert(errorCode, __FILE__, __LINE__);
  } catch (int code) {
    REQUIRE(code == errorCode);
    REQUIRE(buffer.str().find("CUDA Error:") != std::string::npos);
  }

  // Restore stderr
  std::cerr.rdbuf(old);
}