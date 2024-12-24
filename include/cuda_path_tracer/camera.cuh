#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <cmath>
#include <driver_types.h>
#include <memory>

//  number of samples for each pixels, used for eantialiasing
#define NUM_SAMPLES 512

class Camera {

public:
  __host__ Camera();
  __host__ Camera(const Vec3 &origin);

  __host__ void render(const std::shared_ptr<Scene> &scene, uchar4 *image);
  __host__ void init(const std::shared_ptr<Scene> &scene);

  __host__ __device__ auto getScene() -> Scene *;
  __host__ __device__ auto getOrigin() -> Vec3;
  __host__ __device__ auto getViewportWidth() -> float;
  __host__ __device__ auto getViewportHeight() -> float;

private:
  /**
   * @brief viewportWidth and viewportHeight are the dimensions of the camera's
   * view. The viewport is a rectangle that represents the camera's view (our
   * world coordinates).
   */
  float viewportWidth, viewportHeight = 2.0f; // NOLINT

  float verticalFov = 20.0f; // NOLINT

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
  Vec3 origin = {0, 0, 0};

  /**
   * @brief The point the camera is looking at.
   */
  Vec3 lookAt = {0, 0, -1};

  /**
   * @brief The camera's up vector.
   */
  Vec3 up = {0, 1, 0};

  Vec3 defocusDiskU, defocusDiskV;

  float defocusAngle = 10, focusDistance = 3.4; // TODO: make these configurable
};
