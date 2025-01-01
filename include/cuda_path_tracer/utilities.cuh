#pragma once

#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#define DEGREE_TO_RADIAN(deg) (deg * M_PIf32 / 180.0f)

// TODO: This should be replaced with the official universal_host_pinned_vector
// implemented in https://github.com/NVIDIA/cccl/pull/2653 once a new release is
// available

using mr = thrust::universal_host_pinned_memory_resource;

template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, mr>;

template <typename T>
using universal_host_pinned_vector =
    thrust::host_vector<T, pinned_allocator<T>>;
