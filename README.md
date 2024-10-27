# CUDA-Path-Tracer
Project for the course in GPU Computing at the University of Grenoble Alpes, INP ENSIMAG

## Running

To run the project use the following command:

```bash
cmake -S . -B build
cmake --build build
./build/apps/cuda_path_tracer
```

## Testing

<!-- FIXME: Broken linking -->

To run tests use the following command:

```bash
cmake --build build --target tests
ctest
```