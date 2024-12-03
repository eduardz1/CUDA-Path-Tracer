#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>
#include <memory>

class Camera {

public:
  __host__ __device__ Camera();
  __host__  __device__ Camera(const Vec3 &origin);

  __host__ void render(const std::shared_ptr<Scene> &scene, uchar4 *image);
  __device__ auto getRay(const uint16_t x, const uint16_t y) -> Ray;
  __device__ auto getColor(const Ray &ray) -> uchar4;

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
};
