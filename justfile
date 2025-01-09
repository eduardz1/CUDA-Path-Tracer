type      := "Release"
build_dir := "build"
bench_dir := "bench"
target    := build_dir / "apps/cuda_path_tracer"

# Lists all available targets
@default:
    just --list

# Builds the application with the specified CMake arguments
[positional-arguments]
@build *args='':
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE={{type}} $@
    cmake --build {{build_dir}}

# Builds the application with testing enabled
@test:
    just build -DBUILD_TESTING=ON
    ctest

# Runs the application
@run: build
    ./{{target}}

# Benchmarks the application using NVIDIA Nsight Systems
@bench: build
    mkdir -p {{bench_dir}}
    rm -rf {{bench_dir}}/*
    nsys profile --stats=true -o {{bench_dir}}/bench ./{{target}}
    nsys analyze {{bench_dir}}/bench.sqlite

# Cleans the build and benchmark directories, as well as any generated images
@clean:
    rm -rf {{build_dir}} {{bench_dir}} *.ppm
