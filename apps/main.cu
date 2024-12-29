
#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector_functions.h>
#include <vector_types.h>

void demo3Spheres(const uint16_t image_width, const uint16_t image_height,
                  thrust::host_vector<uchar4> &image) {
  const auto shapes = thrust::device_vector<Shape>{
      Sphere{{0, 0, -1.2}, 0.5}, Sphere{{-1, 0, -1}, 0.5},
      Sphere{{1, 0, -1}, 0.5}, Sphere{{0, -100.5, -1}, 100}};
  const auto scene = std::make_shared<Scene>(image_width, image_height, shapes);

  auto camera = CameraBuilder()
                    .origin({-2, 2, 1})
                    .lookAt({0, 0, -1})
                    .up({0, 1, 0})
                    .defocusAngle(10)   // NOLINT
                    .focusDistance(3.4) // NOLINT
                    .verticalFov(20.0f) // NOLINT
                    .build();
  camera.render(scene, image);
}

void cornellBox(const uint16_t image_width, const uint16_t image_height,
                thrust::host_vector<uchar4> &image) {
  const auto shapes = thrust::device_vector<Shape>{
      Parallelogram{{555, 0, 0}, {0, 555, 0}, {0, 0, 555}},       // left wall
      Parallelogram{{0, 0, 0}, {0, 555, 0}, {0, 0, 555}},         // right wall
      Parallelogram{{0, 0, 0}, {555, 0, 0}, {0, 0, 555}},         // floor
      Parallelogram{{0, 555, 0}, {555, 0, 0}, {0, 0, 555}},       // ceiling
      Parallelogram{{0, 0, 555}, {555, 0, 0}, {0, 555, 0}},       // back wall
      Parallelogram{{343, 554, 332}, {-130, 0, 0}, {0, 0, -105}}, // light
      RectangularCuboid{{130, 0, 65}, {295, 165, 230}}
          .rotate({0, -15, 0})
          .translate({-40, 0, -20}),
      RectangularCuboid{{265, 0, 295}, {430, 330, 460}}
          .rotate({0, 18, 0})
          .translate({120, 0, 60}),
  };
  const auto scene = std::make_shared<Scene>(image_width, image_height, shapes);

  auto camera = CameraBuilder()
                    .origin({278, 278, -800}) // NOLINT
                    .lookAt({278, 278, 0})    // NOLINT
                    .up({0, 1, 0})
                    .defocusAngle(0)     // NOLINT
                    .focusDistance(10.0) // NOLINT
                    .verticalFov(40.0f)  // NOLINT
                    .build();
  camera.render(scene, image);
}

auto main() -> int {
  constexpr auto image_width = 512;
  constexpr auto image_height = 512;
  constexpr auto num_pixels = image_width * image_height;

  thrust::host_vector<uchar4> image(num_pixels);

  cornellBox(image_width, image_height, image);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image);

  return EXIT_SUCCESS;
}
