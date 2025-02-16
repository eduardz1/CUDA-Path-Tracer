#include "cuda_path_tracer/project.cuh"
#include <CLI/CLI.hpp>

auto main(int argc, char **argv) -> int {
  CLI::App app{"CUDA Path Tracer"};
  argv = app.ensure_utf8(argv);

  std::string scene_file;
  std::string quality;

  app.add_option("-s,--scene", scene_file, "Scene file")->required();
  app.add_option("-q,--quality", quality, "Quality of the render")
      ->default_val("high")
      ->check(CLI::IsMember({"low", "medium", "high"}));

  CLI11_PARSE(app, argc, argv);

  if (scene_file.empty()) {
    std::cerr << "Scene file is required" << '\n';
    return EXIT_FAILURE;
  }

  Project::load(scene_file, quality)->render();

  return EXIT_SUCCESS;
}
