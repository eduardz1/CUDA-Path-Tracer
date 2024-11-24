#pragma once

#include "cuda_path_tracer/scene.cuh"
#include "cuda_path_tracer/vec3.cuh"
#include <driver_types.h>
#include <memory>

class Camera {

public:
  __host__ Camera();
  __host__ Camera(const Vec3 &origin, std::shared_ptr<Scene> scene);

  __host__ void render() const;

private:
  /**
   * @brief The scene to render
   */
  std::shared_ptr<Scene> scene;

  /**
   * @brief viewportWidth and viewportHeight are the dimensions of the camera's
   * view. The viewport is a rectangle that represents the camera's view (our
   * world coordinates).
   */
  float viewportWidth, viewportHeight = 2.0f; // NOLINT

  /**
   * @brief The X Y Z coordinates of the camera's origin.
   */
  Vec3 origin;
};
