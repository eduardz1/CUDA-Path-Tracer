#include "cuda_path_tracer/color.cuh"
#include "cuda_path_tracer/image.cuh"
#include "cuda_path_tracer/project.cuh"
#include <exception>
#include <fstream>
#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>

namespace {
// Compile time string hashing
constexpr auto hash_string(std::string_view str) -> uint64_t {
  uint64_t hash = 14695981039346656037ULL; // NOLINT
  for (char c : str) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL; // NOLINT
  }
  return hash;
}

constexpr auto operator""_hash(const char *str, size_t /*unused*/) -> uint64_t {
  return hash_string(str);
}

__host__ auto parseColor(const nlohmann::json &j) -> Color {
  // If it's an array of integers, parse as RGB values, if they are floats, as
  // normalized RGB values
  if (j.is_array()) {
    if (j[0].is_number_integer()) {
      return Color::RGB(j[0], j[1], j[2]);
    }
    return Color::Normalized(j[0], j[1], j[2]);
  }

  // If it's a string, parse as color name
  const auto colorName = j.get<std::string>();

  switch (hash_string(colorName)) {
  case "white"_hash: return Colors::White;
  case "black"_hash: return Colors::Black;
  case "gray"_hash: return Colors::Gray;
  case "red"_hash: return Colors::Red;
  case "green"_hash: return Colors::Green;
  case "blue"_hash: return Colors::Blue;
  case "yellow"_hash: return Colors::Yellow;
  case "cyan"_hash: return Colors::Cyan;
  case "magenta"_hash: return Colors::Magenta;

  case "rosewater"_hash: return Catpuccin::Latte::Rosewater;
  case "flamingo"_hash: return Catpuccin::Latte::Flamingo;
  case "pink"_hash: return Catpuccin::Latte::Pink;
  case "mauve"_hash: return Catpuccin::Latte::Mauve;
  case "latte-red"_hash: return Catpuccin::Latte::Red;
  case "maroon"_hash: return Catpuccin::Latte::Maroon;
  case "peach"_hash: return Catpuccin::Latte::Peach;
  case "latte-yellow"_hash: return Catpuccin::Latte::Yellow;
  case "latte-green"_hash: return Catpuccin::Latte::Green;
  case "teal"_hash: return Catpuccin::Latte::Teal;
  case "sky"_hash: return Catpuccin::Latte::Sky;
  case "sapphire"_hash: return Catpuccin::Latte::Sapphire;
  case "latte-blue"_hash: return Catpuccin::Latte::Blue;
  case "lavander"_hash: return Catpuccin::Latte::Lavander;
  case "text"_hash: return Catpuccin::Latte::Text;
  case "base"_hash: return Catpuccin::Latte::Base;
  case "mantle"_hash: return Catpuccin::Latte::Mantle;
  case "crust"_hash: return Catpuccin::Latte::Crust;

  default: throw std::runtime_error("Unknown color name: " + colorName);
  }
}

__host__ auto parseVec3(const nlohmann::json &j) -> Vec3 {
  return Vec3{j[0], j[1], j[2]};
}

__host__ auto parseTexture(const nlohmann::json &j) -> Texture {
  if (j.contains("checker")) {
    return Checker(j["checker"]["scale"].get<float>(),
                   parseColor(j["checker"]["even"]),
                   parseColor(j["checker"]["odd"]));
  }

  return parseColor(j);
}

__host__ auto parseMaterial(const nlohmann::json &j) -> Material {
  const auto type = j["type"].get<std::string>();

  switch (hash_string(type)) {
  case "lambertian"_hash: {
    if (j.contains("texture")) {
      return Lambertian(parseTexture(j["texture"]));
    }
    return Lambertian(parseColor(j["color"]));
  }
  case "dielectric"_hash: return Dielectric(j["refraction_index"].get<float>());
  case "metal"_hash:
    return Metal(parseColor(j["color"]), j["fuzz"].get<float>());
  case "light"_hash: return Light(Color{parseVec3(j["color"])});

  default: throw std::runtime_error("Unknown material type: " + type);
  }
}

__host__ auto parseShape(const nlohmann::json &j) -> Shape {
  const auto type = j["type"].get<std::string>();

  switch (hash_string(type)) {
  case "sphere"_hash:
    return Sphere{parseVec3(j["center"]), j["radius"].get<float>(),
                  parseMaterial(j["material"])};
  case "rectangular_cuboid"_hash: {
    auto r = RectangularCuboid{parseVec3(j["vertices"][0]),
                               parseVec3(j["vertices"][1]),
                               parseMaterial(j["material"])};

    if (j.contains("rotation")) {
      r = r.rotate(parseVec3(j["rotation"]));
    }
    if (j.contains("translation")) {
      r = r.translate(parseVec3(j["translation"]));
    }

    return r;
  }
  case "parallelogram"_hash:
    return Parallelogram{parseVec3(j["origin"]), parseVec3(j["u"]),
                         parseVec3(j["v"]), parseMaterial(j["material"])};

  default: throw std::runtime_error("Unknown shape type: " + type);
  }
}

template <typename QualityType>
__host__ auto
buildCamera(const nlohmann::json &cam) -> std::shared_ptr<Camera<QualityType>> {
  auto camera_builder = CameraBuilder<QualityType>();

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
  if (cam.contains("background")) {
    camera_builder.background(parseColor(cam["background"]));
  }

  return std::make_shared<Camera<QualityType>>(camera_builder.build());
}

__host__ auto validateJsonSchema(const nlohmann::json &j) {
  nlohmann::json_schema::json_validator validator;

  try {
    validator.set_root_schema(schema);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error setting root schema: " +
                             std::string(e.what()));
  }

  try {
    validator.validate(j);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error validating JSON: " + std::string(e.what()));
  }
}

__host__ auto loadJsonFromFile(const std::string &filename) -> nlohmann::json {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  try {
    return nlohmann::json::parse(file);
  } catch (const std::exception &e) {
    throw std::runtime_error("Error parsing JSON: " + std::string(e.what()));
  }
}
} // namespace

__host__ auto
Project::load(const std::string &filename,
              const std::string &quality) -> std::shared_ptr<Project> {
  auto j = loadJsonFromFile(filename);
  validateJsonSchema(j);

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

  // Set camera based on quality
  const auto &cam = j["camera"];
  if (quality == "low") {
    project->camera = buildCamera<LowQuality>(cam);
  } else if (quality == "medium") {
    project->camera = buildCamera<MediumQuality>(cam);
  } else if (quality == "high") {
    project->camera = buildCamera<HighQuality>(cam);
  }

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
