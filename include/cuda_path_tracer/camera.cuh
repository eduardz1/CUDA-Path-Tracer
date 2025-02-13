#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

// number of samples for each pixels, used for eantialiasing
#define NUM_SAMPLES 64

// number of images to render and then average together, provides a similar
// effect to antialiasing (controlled by NUM_SAMPLES) but it's more
// parallelizable ideally try to balance the two hyper parameters. They are
// pretty much the same thing, so 16x32 is the same as 32x16.
#define NUM_IMAGES 16

// If true, the images will be averaged using thrust, otherwise they will be
// averaged using a custom kernel.
#define AVERAGE_WITH_THRUST true

class Camera {
public:
  friend class CameraBuilder;

  __host__ Camera();
  __host__ void render(const std::shared_ptr<Scene> &scene,
                       thrust::universal_host_pinned_vector<uchar4> &image);
  __host__ static void
  averageRenderedImages(thrust::universal_host_pinned_vector<uchar4> &output,
                        const cuda::std::span<Vec3> &images,
                        const uint16_t width, const uint16_t height,
                        const uint16_t padded_width);

private:
  __host__ void init(const std::shared_ptr<Scene> &scene);

  /**
   * @brief viewportWidth and viewportHeight are the dimensions of the camera's
   * view. The viewport is a rectangle that represents the camera's view (our
   * world coordinates).
   */
  float viewportWidth, viewportHeight = 2.0F;

  /**
   * @brief They are, respectively, the horizontal and vertical vectors of the
   * distance between each pixel center in world coordinates.
   */
  Vec3 deltaU, deltaV;

  /**
   * @brief Location of the center of the pixel at the top left corner of the
   * image.
   */
  Vec3 pixel00;

  /**
   * @brief The X Y Z coordinates of the camera's origin.
   */
  Vec3 origin;

  /**
   * @brief The point the camera is looking at.
   */
  Vec3 lookAt;

  /**
   * @brief The camera's up vector.
   */
  Vec3 up;

  Vec3 defocusDiskU, defocusDiskV;

  float defocusAngle, focusDistance, verticalFov;

  Vec3 background;
};

class CameraBuilder {
public:
  __host__ CameraBuilder();
  __host__ auto origin(const Vec3 &origin) -> CameraBuilder &;
  __host__ auto lookAt(const Vec3 &lookAt) -> CameraBuilder &;
  __host__ auto up(const Vec3 &up) -> CameraBuilder &;
  __host__ auto verticalFov(float verticalFov) -> CameraBuilder &;
  __host__ auto focusDistance(float focusDistance) -> CameraBuilder &;
  __host__ auto defocusAngle(float defocusAngle) -> CameraBuilder &;
  __host__ auto background(const Vec3 &background) -> CameraBuilder &;
  __host__ auto build() -> Camera;

private:
  Camera camera;
};