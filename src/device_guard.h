/**
 * @file device_guard.h
 * @brief selected_backend.h + the "must be a device pass" guard, in one include
 *
 * The six device headers here (runtime.cuh, complex.cuh, fp16.cuh,
 * bf16.cuh, fp8.cuh, rand.cuh) open with the same two requirements: the selected
 * backend, and a refusal to compile outside a device pass. Factored here so the
 * guard exists once.
 *
 * The two questions stay apart, as they must: "which
 * backend?" is selected_backend.h's, answered from the compiler's device macro
 * or the CMake define; "is this a device pass?" is this guard's, answered from
 * the compiler's device macro alone. selected_backend.h resolves a backend in a
 * host compile too, so the guard cannot be folded into it -- and must not be,
 * because the bridges include selected_backend.h from host compiles and must
 * NOT #error there. The guard is device-only, so it lives one layer up, here.
 *
 * See src/README.md ("The switch points").
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP.
#include "selected_backend.h"

#if !defined(__CUDACC__) && !defined(__HIP__) && !defined(__HIPCC__)
#error                                                                                             \
    "gpu device header included outside a CUDA or HIP device-compile pass (__CUDACC__ / __HIP__ / __HIPCC__ not defined)"
#endif
