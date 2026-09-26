/**
 * @file selected_backend.h
 * @brief Which backend was selected, from whatever evidence the TU has
 *
 * Defines exactly one of GPUMOD_SELECTED_CUDA / GPUMOD_SELECTED_HIP, and
 * nothing else -- no type, no function, no vendor header. That is why it is
 * not a *_bridge.h despite reaching both host compiles and device passes: it
 * carries no declaration across that boundary.
 *
 * The ladder reads the compiler's device macros first, so a device pass never
 * depends on a CMake define, then falls back to GPUMOD_GPU_BACKEND_* for a
 * host compile (link gpumod_backend PRIVATE to get it).
 *
 * The legitimate readers are the switch points that include it: device_guard.h
 * directly, and the four .cuh headers transitively through it. The two bridges
 * (gpu_stream_bridge.h and rand_state_bridge.h, now in src/extension/bridge/)
 * include it directly too -- directly rather than through device_guard.h,
 * because a bridge compiles in a host TU and so must not carry its device-pass
 * #error. Linking gpumod_backend grants the ability to write a backend #if
 * above src and is not a licence to -- see src/README.md.
 */

#pragma once

#if defined(__CUDACC__)
#define GPUMOD_SELECTED_CUDA 1
#elif defined(__HIP__) || defined(__HIPCC__)
#define GPUMOD_SELECTED_HIP 1
#elif defined(GPUMOD_GPU_BACKEND_CUDA)
#define GPUMOD_SELECTED_CUDA 1
#elif defined(GPUMOD_GPU_BACKEND_HIP)
#define GPUMOD_SELECTED_HIP 1
#else
#error                                                                                             \
    "gpu/selected_backend.h: no backend selected -- link gpumod_backend PRIVATE (host compile), or include from a device pass"
#endif
