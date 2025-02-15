#pragma once

#include "cuda_path_tracer/camera.cuh"
#include "cuda_path_tracer/scene.cuh"
#include <memory>
#include <string>

class Project {
public:
  __host__ static auto
  load(const std::string &filename) -> std::shared_ptr<Project>;
  __host__ auto render() -> void;

private:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Camera<>> camera;
  std::string name;
};