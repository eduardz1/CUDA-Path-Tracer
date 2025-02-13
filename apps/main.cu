#include "cuda_path_tracer/project.cuh"
#include <CLI/CLI.hpp>

// TODO(eduard): cornell_box.json is missing the back wall but it looks kinda
// nice like this
auto main(int argc, char **argv) -> int {
  CLI::App app{"CUDA Path Tracer"};

  std::string scene_file;
  std::string output_file = "output.ppm";

  app.add_option("-s,--scene", scene_file, "Scene file")->required();

  CLI11_PARSE(app, argc, argv);

  if (scene_file.empty()) {
    std::cerr << "Scene file is required" << std::endl;
    return EXIT_FAILURE;
  }

  const auto project = Project::load(scene_file);
  project->render();

  return EXIT_SUCCESS;
}
