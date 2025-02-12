#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/error.cuh"
#include "cuda_path_tracer/materials/lambertian.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace {
template <dim3 BlockSize, uint16_t NumSamples, uint16_t NumImages,
          bool AverageWithThrust, typename State>
auto createCamera() {
  return CameraBuilder<BlockSize, NumSamples, NumImages, AverageWithThrust,
                       State>()
      .origin({-2, 2, 1})
      .lookAt({0, 0, -1})
      .up({0, 1, 0})
      .defocusAngle(10)
      .focusDistance(3.4)
      .verticalFov(20.0F)
      .build();
}

template <uint32_t BlockSizeX, uint32_t BlockSizeY, uint16_t NumSamples,
          uint16_t NumImages, bool AverageWithThrust, typename State>
void runBenchmark(const std::shared_ptr<Scene> &scene,
                  thrust::universal_host_pinned_vector<uchar4> &image) {
  constexpr dim3 BlockSize{BlockSizeX, BlockSizeY};
  auto camera = createCamera<BlockSize, NumSamples, NumImages,
                             AverageWithThrust, State>();

  std::string config_name =
      std::string(std::is_same_v<State, curandStatePhilox4_32_10_t>
                      ? "Philox"
                      : "curandState_t") +
      " " + std::to_string(BlockSize.x) + "x" + std::to_string(BlockSize.y) +
      " block, " + std::to_string(NumSamples) + "/" +
      std::to_string(NumImages) + (AverageWithThrust ? " thrust" : " custom") +
      " averaging";

  BENCHMARK(config_name.c_str()) {
    camera.render(scene, image);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return image[0].x;
  };
}

template <typename State>
void runAllCombinations(const std::shared_ptr<Scene> &scene,
                        thrust::universal_host_pinned_vector<uchar4> &image) {
  // 4x4 blocks
  runBenchmark<4, 4, 16, 256, true, State>(scene, image);
  runBenchmark<4, 4, 16, 256, false, State>(scene, image);
  runBenchmark<4, 4, 32, 128, true, State>(scene, image);
  runBenchmark<4, 4, 32, 128, false, State>(scene, image);
  runBenchmark<4, 4, 64, 64, true, State>(scene, image);
  runBenchmark<4, 4, 64, 64, false, State>(scene, image);

  // 8x8 blocks
  runBenchmark<8, 8, 16, 256, true, State>(scene, image);
  runBenchmark<8, 8, 16, 256, false, State>(scene, image);
  runBenchmark<8, 8, 32, 128, true, State>(scene, image);
  runBenchmark<8, 8, 32, 128, false, State>(scene, image);
  runBenchmark<8, 8, 64, 64, true, State>(scene, image);
  runBenchmark<8, 8, 64, 64, false, State>(scene, image);

  // 16x16 blocks
  runBenchmark<16, 16, 16, 256, true, State>(scene, image);
  runBenchmark<16, 16, 16, 256, false, State>(scene, image);
  runBenchmark<16, 16, 32, 128, true, State>(scene, image);
  runBenchmark<16, 16, 32, 128, false, State>(scene, image);
  runBenchmark<16, 16, 64, 64, true, State>(scene, image);
  runBenchmark<16, 16, 64, 64, false, State>(scene, image);

  // 32x8 blocks
  runBenchmark<32, 8, 16, 256, true, State>(scene, image);
  runBenchmark<32, 8, 16, 256, false, State>(scene, image);
  runBenchmark<32, 8, 32, 128, true, State>(scene, image);
  runBenchmark<32, 8, 32, 128, false, State>(scene, image);
  runBenchmark<32, 8, 64, 64, true, State>(scene, image);
  runBenchmark<32, 8, 64, 64, false, State>(scene, image);

  // 8x32 blocks
  runBenchmark<8, 32, 16, 256, true, State>(scene, image);
  runBenchmark<8, 32, 16, 256, false, State>(scene, image);
  runBenchmark<8, 32, 32, 128, true, State>(scene, image);
  runBenchmark<8, 32, 32, 128, false, State>(scene, image);
  runBenchmark<8, 32, 64, 64, true, State>(scene, image);
  runBenchmark<8, 32, 64, 64, false, State>(scene, image);

  // 32x32 blocks
  runBenchmark<32, 32, 16, 256, true, State>(scene, image);
  runBenchmark<32, 32, 16, 256, false, State>(scene, image);
  runBenchmark<32, 32, 32, 128, true, State>(scene, image);
  runBenchmark<32, 32, 32, 128, false, State>(scene, image);
  runBenchmark<32, 32, 64, 64, true, State>(scene, image);
  runBenchmark<32, 32, 64, 64, false, State>(scene, image);
}

auto createTestScene(uint16_t width,
                     uint16_t height) -> std::shared_ptr<Scene> {
  const auto shapes = thrust::device_vector<Shape>{
      Sphere{{0, 0, -1.2}, 0.5, Lambertian(Vec3{0.7, 0.3, 0.3})},
      Sphere{{-1, 0, -1}, 0.5, Dielectric(1.50)},
      Sphere{{1, 0, -1}, 0.5, Dielectric(1.00 / 1.50)},
      Sphere{{0, -100.5, -1}, 100, Lambertian(Vec3{0.8, 0.8, 0.0})},
      RectangularCuboid{{130, 0, 65}, {295, 165, 230}}
          .rotate({0, -15, 0})
          .translate({40, 0, -20})};
  return std::make_shared<Scene>(width, height, shapes);
}

auto createCamera() {
  return CameraBuilder()
      .origin({-2, 2, 1})
      .lookAt({0, 0, -1})
      .up({0, 1, 0})
      .defocusAngle(10)
      .focusDistance(3.4)
      .verticalFov(20.0F)
      .build();
}
} // namespace

TEST_CASE("Hyperparameter Optimization", "[benchmark]") {
  constexpr uint16_t width = 512;
  constexpr uint16_t height = 512;
  constexpr auto num_pixels = static_cast<size_t>(width * height);

  auto scene = createTestScene(width, height);
  thrust::universal_host_pinned_vector<uchar4> image(num_pixels);

  runAllCombinations<curandState_t>(scene, image);
  runAllCombinations<curandStatePhilox4_32_10_t>(scene, image);
}

TEST_CASE("Render Output Validation", "[render]") {
  constexpr uint16_t width = 64;
  constexpr uint16_t height = 64;
  constexpr auto num_pixels = static_cast<size_t>(width * height);

  auto scene = createTestScene(width, height);
  auto camera = createCamera();

  thrust::universal_host_pinned_vector<uchar4> image1(num_pixels);
  thrust::universal_host_pinned_vector<uchar4> image2(num_pixels);

// Render with curandState_t
#undef USE_PHILOX
  camera.render(scene, image1);

// Render with Philox
#define USE_PHILOX
  camera.render(scene, image2);

  // Verify outputs are reasonably similar
  float max_diff = 0.0F;
  for (size_t i = 0; i < num_pixels; ++i) {
    float diff = std::abs(static_cast<float>(image1[i].x) -
                          static_cast<float>(image2[i].x)) +
                 std::abs(static_cast<float>(image1[i].y) -
                          static_cast<float>(image2[i].y)) +
                 std::abs(static_cast<float>(image1[i].z) -
                          static_cast<float>(image2[i].z));
    max_diff = std::max(max_diff, diff);
  }

  // Allow some variation due to different RNG implementations
  REQUIRE(max_diff < 50.0F);
}

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)