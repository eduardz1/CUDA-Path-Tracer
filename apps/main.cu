#include "cuda_path_tracer/project.cuh"
#include <CLI/CLI.hpp>

// TODO(eduard): cornell_box.json is missing the back wall but it looks kinda
// nice like this
auto main(int argc, char **argv) -> int {
  CLI::App app{"CUDA Path Tracer"};

  std::string scene_file;

  app.add_option("-s,--scene", scene_file, "Scene file")->required();

  CLI11_PARSE(app, argc, argv);

  if (scene_file.empty()) {
    std::cerr << "Scene file is required" << '\n';
    return EXIT_FAILURE;
  }

  Project::load(scene_file)->render();

  return EXIT_SUCCESS;
}
