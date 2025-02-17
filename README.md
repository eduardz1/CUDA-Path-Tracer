# CUDA-Path-Tracer

## Commands

If you have the [just](https://github.com/casey/just) command runner installed you can run the command `just` to see the list of available commands. In particular, to run the application use:

```bash
just run -s examples/cornell_box.json
```

or equivalently:

```bash
cmake -S . -B build
cmake --build build
./build/apps/cuda_path_tracer -s examples/cornell_box.json
```

Where the `-s` flag (or `--scene`) specifies the scene file to render. You can find example scene files in the `examples/` directory. You can also specify the quality of the render using the `-q` flag (or `--quality`), for example:

```bash
just run -s examples/cornell_box.json -q high
```

Where "high" is the default quality level. The available quality levels are "low", "medium" and "high".

Each argument passed to the `just build` command is forwarded to `cmake`, for example, to build the project for all major CUDA architectures use:

```bash
just build -DCMAKE_CUDA_ARCHITECTURES=all-major
```

### Testing and Benchmarking

Benchmarks are integrated in the testing suite. The `just bench` command will run NVIDIA's Nsight Systems profiler on the main application. To run the tests with the integrated benchmarks use:

```bash
just test
```

or equivalently:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
./build/tests/tests
```

> [!TIP]
> Certain benchmarks in the test suite will take a while to complete, to run only the unit tests you can use the `just test --skip-benchmarks` command.

## Repository Structure

### Code

- [`apps/`](apps/): Application code that uses the `cuda_path_tracer` library. Here you can find a demo application that renders a scene using the library.
- [`examples/`](examples/): Example scene files that can be rendered using the demo application.
- [`include/`](include/): Public headers of the `cuda_path_tracer` library. Here you can find the `Camera` class that is used to render a scene. The core of the library is in [include/cuda_path_tracer/camera.cuh](include/cuda_path_tracer/camera.cuh) and its inline implementation in [include/cuda_path_tracer/camera.inl](include/cuda_path_tracer/camera.inl) due to the camera code making heavy usage of templating to handle hyperparameters.
- [`src/`](src/): Implementation of the `cuda_path_tracer` library.
- [`tests/`](tests/): Unit tests for the `cuda_path_tracer` library and benchmarks. In particular, in [`tests/test_render_bench.cu](tests/test_render_bench.cu) you can find benchmarks for hyperparameters tuning of the `Camera` class and an example of direct usage of the `cuda_path_tracer` library.

### Report

- [`report/`](report/): Typst source code of the report.
