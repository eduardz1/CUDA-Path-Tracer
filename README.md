# CUDA-Path-Tracer

Project for the course in GPU Computing at the University of Grenoble Alpes, INP ENSIMAG.

> [!CAUTION]
> The minimum required version of the CUDA Toolkit is 12.4.

## Running

To run the project use the following command:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/apps/cuda_path_tracer
```

## Testing

<!-- FIXME: Broken linking -->

To run tests use the following command:

```bash
cmake --build build --target test
ctest
```