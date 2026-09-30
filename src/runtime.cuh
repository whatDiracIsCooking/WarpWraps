/**
 * @file runtime.cuh
 * @brief The CUDA-or-HIP device-compile macros for .cu translation units
 *
 * #included directly into a .cu or -x hip device-compiled TU, which imports no
 * modules and so cannot use backend.h. The backend comes from the
 * compiler's own device-compile macro; #errors outside a device pass. Link
 * wwr.device for the include path and the runtime headers.
 *
 *   WWR_GRID_CONSTANT   __grid_constant__ under CUDA, empty under HIP
 *   WWR_WARP_SIZE       warp/wavefront size, as a constant expression. Set
 *                       with -DWWR_WARP_SIZE (default 32; 64 for CDNA)
 *
 * wwrStream_t comes from runtime.h, #included below -- the same vendor handle
 * wwr.runtime_api exports to importers. The full runtime headers here are a
 * superset of the minimal one runtime.h pulls.
 *
 * Nothing here is #undef'd; include it once, near the top of a device TU.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass.
#include "device_guard.h"

// wwrStream_t -- the same vendor handle wwr.runtime_api exports, so a device TU
// and a host-created stream name one type. Its minimal vendor header is subsumed
// by the full runtime pulled in below.
#include "runtime.h"

// Past the guard, selected_backend.h's ladder took its answer from the same
// compiler macro, so "device pass?" and "which backend?" cannot disagree.
#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime.h>

#define WWR_GRID_CONSTANT __grid_constant__

#else

#include <hip/hip_runtime.h>

#define WWR_GRID_CONSTANT

#endif

// Backend-independent, and identical in both of HIP's compile passes -- which
// neither backend's own warp-size spelling is.
#ifndef WWR_WARP_SIZE
#define WWR_WARP_SIZE 32
#endif

static_assert(WWR_WARP_SIZE == 32 || WWR_WARP_SIZE == 64,
              "WWR_WARP_SIZE must be 32 or 64 "
              "(32 for NVIDIA and RDNA, 64 for CDNA)");
