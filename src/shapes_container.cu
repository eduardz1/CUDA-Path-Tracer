// #include "cuda_path_tracer/error.cuh"
// #include "cuda_path_tracer/shapes_container.cuh"
// #include "cuda_path_tracer/sphere.cuh"

// // __host__ __device__ ShapesContainer::ShapesContainer() {};

// __host__ __device__ auto ShapesContainer::getShapeAt(size_t idx) -> char * {
//   return shapes + offsets[idx];
// };

// __host__ __device__ auto
// ShapesContainer::getShapeTypeAt(size_t idx) -> ShapeType {
//   return shapeTypes[idx];
// };
// __host__ __device__ auto ShapesContainer::getNumShapes() -> uint16_t {
//   return numShapes;
// };

// __host__ auto ShapesContainer::copyShapesToDevice(
//     const std::vector<Shape *> &h_shapes) -> void {

//   numShapes = h_shapes.size();
//   size_t h_size = 0;
//   std::vector<size_t> h_offsets;
//   std::vector<ShapeType> h_types;
//   for (auto shape : h_shapes) {
//     h_offsets.push_back(h_size);
//     h_types.push_back(shape->getShapeType());
//     switch (shape->getShapeType()) {
//     case ShapeType::SPHERE:
//       h_size += sizeof(Sphere);
//       break;
//     default:
//       printf("error!!\n");
//     }
//   }

//   printf("h_size: %zu\n", h_size);
//   printf("len offsets: %zu\n", h_types.size());
//   printf("len types: %zu\n", h_types.size());

//   CUDA_ERROR_CHECK(cudaMalloc(&shapes, h_size));
//   printf("malloc 1");
//   CUDA_ERROR_CHECK(cudaMalloc(&offsets, h_offsets.size() * sizeof(size_t)));
//   printf("malloc 2");
//   CUDA_ERROR_CHECK(cudaMalloc(&shapeTypes, h_types.size() * sizeof(ShapeType)));
//   printf("malloc 3");
//   CUDA_ERROR_CHECK(cudaGetLastError());

//   printf("po mallocach");
//   for (size_t i = 0; i < numShapes; ++i) {
//     cudaMemcpy(shapes + offsets[i], h_shapes[i], sizeof(*h_shapes[i]),
//                cudaMemcpyHostToDevice);
//   }
//   CUDA_ERROR_CHECK(cudaMemcpy(offsets, h_offsets.data(),
//                               h_offsets.size() * sizeof(size_t),
//                               cudaMemcpyHostToDevice));
//   CUDA_ERROR_CHECK(cudaMemcpy(shapeTypes, h_types.data(),
//                               h_types.size() * sizeof(ShapeType),
//                               cudaMemcpyHostToDevice));
//   CUDA_ERROR_CHECK(cudaGetLastError());
//   printf("end of copytodevice");
// };

// // __host__ __device__ ShapesContainer::~ShapesContainer() {
// //   cudaFree(shapes);
// //   cudaFree(offsets);
// //   cudaFree(shapeTypes);
// // }
