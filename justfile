set positional-arguments

export CMAKE_BUILD_TYPE := "Release"

build_dir := "build"
bench_dir := "bench"
target    := build_dir / "apps/cuda_path_tracer"

# Lists all available targets
@default:
    just --list

# Builds the application with the specified CMake arguments
@build *CMAKE_ARGS:
    cmake -S . -B {{build_dir}} $@
    cmake --build {{build_dir}}

# Builds the application with testing enabled
@test *CMAKE_ARGS: (build "-DBUILD_TESTING=ON" "-DCMAKE_BUILD_TYPE=Debug" CMAKE_ARGS)
    # It would be better to have the target be
    # 'test $CMAKE_BUILD_TYPE="Debug" *CMAKE_ARGS: ...' to overload the
    # CMAKE_BUILD_TYPE environment variable, but
    # https://github.com/casey/just/issues/1804 needs to be resolved first.
    ./{{build_dir}}/tests/tests

# Runs the application
@run *CMAKE_ARGS: (build CMAKE_ARGS)
    ./{{target}}

# Benchmarks the application using NVIDIA Nsight Systems
@bench *CMAKE_ARGS: (build CMAKE_ARGS)
    mkdir -p {{bench_dir}}
    rm -rf {{bench_dir}}/*
    nsys profile --stats=true -o {{bench_dir}}/bench ./{{target}}
    nsys analyze {{bench_dir}}/bench.sqlite

# Uses the NVIDIA Compute Sanitizer to check for memory leaks
@memcheck *CMAKE_ARGS: (build CMAKE_ARGS)
    compute-sanitizer --show-backtrace yes --tool memcheck --leak-check full ./{{target}}

# Cleans the build and benchmark directories, as well as any generated images
@clean:
    rm -rf {{build_dir}} {{bench_dir}} *.ppm
