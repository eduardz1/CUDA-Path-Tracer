#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/lambertian.cuh"
#include "cuda_path_tracer/shapes/sphere.cuh"
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace {
void demo3Spheres(const uint16_t image_width, const uint16_t image_height,
                  thrust::universal_host_pinned_vector<uchar4> &image) {
  const auto shapes = thrust::device_vector<Shape>{
      Sphere{{0, 0, -1.2}, 0.5, Lambertian(Catpuccin::Latte::Flamingo)},
      Sphere{{-1, 0, -1}, 0.5, Dielectric(1.50)},
      Sphere{{1, 0, -1}, 0.5, Dielectric(1.00 / 1.50)},
      Sphere{{0, 0.75, -1}, 0.1, Light(Colors::White)},

      Sphere{{0, -100.5, -1}, 100, Lambertian(Checker(0.32, Colors::Black, Colors::White))}};
  const auto scene = std::make_shared<Scene>(image_width, image_height, shapes);

  auto camera = CameraBuilder()
                    .origin({-2, 2, 1})
                    .lookAt({0, 0, -1})
                    .up({0, 1, 0})
                    .defocusAngle(10)   // NOLINT
                    .focusDistance(3.4) // NOLINT
                    .verticalFov(20.0f) // NOLINT
                    .background(Colors::Black)
                    .build();
  camera.render(scene, image);
}

void cornellBox(const uint16_t image_width, const uint16_t image_height,
                thrust::universal_host_pinned_vector<uchar4> &image) {
  const auto shapes = thrust::device_vector<Shape>{
      Parallelogram{{555, 0, 0}, {0, 555, 0}, {0, 0, 555}, Lambertian(Catpuccin::Latte::Sapphire)},       // left wall
      Parallelogram{{0, 0, 0}, {0, 555, 0}, {0, 0, 555}, Lambertian(Catpuccin::Latte::Sapphire)},         // right wall
      Parallelogram{{0, 0, 0}, {555, 0, 0}, {0, 0, 555}, Lambertian(Catpuccin::Latte::Sapphire)},         // floor
      Parallelogram{{0, 555, 0}, {555, 0, 0}, {0, 0, 555}, Lambertian(Catpuccin::Latte::Sapphire)},       // ceiling
      Parallelogram{{0, 0, 555}, {555, 0, 0}, {0, 555, 0}, Lambertian(Catpuccin::Latte::Sapphire)},       // back wall
      Parallelogram{{343, 554, 332}, {-130, 0, 0}, {0, 0, -105}, Light(Colors::White)}, // light
      RectangularCuboid{{130, 0, 65}, {295, 165, 230}, Lambertian(Checker(0.5, Catpuccin::Latte::Peach, Colors::Black))}
          .rotate({0, -15, 0})
          .translate({-40, 0, -20}),
      RectangularCuboid{{265, 0, 295}, {430, 330, 460}, Metal(Catpuccin::Latte::Lavander, 0.7)}
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
                    .background(Colors::Black)
                    .build();
  camera.render(scene, image);
}
} // namespace

auto main() -> int {
  constexpr auto image_width = 512;
  constexpr auto image_height = 512;
  constexpr auto num_pixels = image_width * image_height;

  thrust::universal_host_pinned_vector<uchar4> image(num_pixels);

  // demo3Spheres(image_width, image_height, image);
  cornellBox(image_width, image_height, image);

  saveImageAsPPM("test_image.ppm", image_width, image_height, image);

  return EXIT_SUCCESS;
}
