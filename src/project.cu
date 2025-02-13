#include "cuda_path_tracer/project.cuh"
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
__host__ auto parseVec3(const nlohmann::json &j) -> Vec3 {
  return Vec3{j[0], j[1], j[2]};
}

__host__ auto parseMaterial(const nlohmann::json &j) -> Material {
  const auto type = j["type"].get<std::string>();

  if (type == "lambertian") {
    return Lambertian(parseVec3(j["color"]));
  }
  if (type == "dielectric") {
    return Dielectric(j["refraction_index"].get<float>());
  }
  if (type == "metal") {
    return Metal(parseVec3(j["color"]), j["fuzz"].get<float>());
  }

  throw std::runtime_error("Unknown material type: " + type);
}

__host__ auto parseShape(const nlohmann::json &j) -> Shape {
  const auto type = j["type"].get<std::string>();
  if (type == "sphere") {
    return Sphere{parseVec3(j["center"]), j["radius"].get<float>(),
                  parseMaterial(j["material"])};
  }
  if (type == "rectangular_cuboid") {
    auto r = RectangularCuboid{
        parseVec3(j["vertices"][0]),
        parseVec3(j["vertices"][1]),
    };

    if (j.contains("rotation")) {
      r = r.rotate(parseVec3(j["rotation"]));
    }
    if (j.contains("translation")) {
      r = r.translate(parseVec3(j["translation"]));
    }

    return r;
  }
  if (type == "parallelogram") {
    return Parallelogram{parseVec3(j["origin"]), parseVec3(j["u"]),
                         parseVec3(j["v"])};
  }
  throw std::runtime_error("Unknown shape type: " + type);
}
} // namespace

__host__ auto
Project::load(const std::string &filename) -> std::shared_ptr<Project> {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  nlohmann::json j;
  file >> j;

  auto project = std::make_shared<Project>();
  project->name = j["name"].get<std::string>() + ".ppm";

  // Parse image dimensions
  const auto width = j["image"]["width"].get<uint16_t>();
  const auto height = j["image"]["height"].get<uint16_t>();

  // Parse shapes
  thrust::device_vector<Shape> shapes;
  for (const auto &shape : j["shapes"]) {
    shapes.push_back(parseShape(shape));
  }

  // Create scene
  project->scene = std::make_shared<Scene>(width, height, shapes);

  // Parse camera settings
  const auto &cam = j["camera"];
  auto camera_builder = CameraBuilder();

  if (cam.contains("origin")) {
    camera_builder.origin(parseVec3(cam["origin"]));
  }
  if (cam.contains("lookAt")) {
    camera_builder.lookAt(parseVec3(cam["lookAt"]));
  }
  if (cam.contains("up")) {
    camera_builder.up(parseVec3(cam["up"]));
  }
  if (cam.contains("verticalFov")) {
    camera_builder.verticalFov(cam["verticalFov"].get<float>());
  }
  if (cam.contains("defocusAngle")) {
    camera_builder.defocusAngle(cam["defocusAngle"].get<float>());
  }
  if (cam.contains("focusDistance")) {
    camera_builder.focusDistance(cam["focusDistance"].get<float>());
  }
  project->camera = std::make_shared<Camera<>>(camera_builder.build());

  return project;
}

__host__ auto Project::render() -> void {
  const auto width = scene->getWidth();
  const auto height = scene->getHeight();
  thrust::universal_host_pinned_vector<uchar4> image(
      static_cast<size_t>(width * height));

  camera->render(scene, image);

  saveImageAsPPM(this->name, width, height, image);
}
