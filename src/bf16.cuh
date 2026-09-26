/**
 * @file bf16.cuh
 * @brief bfloat16 type and float conversions for device-compiled TUs
 *
 * The device-compile counterpart to bf16.cppm: gpuBfloat16 and the
 * float<->bfloat16 conversions. Link gpumod.device. Companion to fp16.cuh.
 *
 * The type is the SAME one bf16.cppm exports under this name, so a host-allocated
 * buffer and a kernel parameter named here agree, and an extern template declared
 * in a .cppm links against a definition compiled in a .cu.
 *
 * Only the conversions are wrapped. bfloat16 carries its arithmetic operators on
 * both backends, so `x + y` on two gpuBfloat16 values is already backend-neutral
 * code naming no vendor symbol -- there is nothing for a wrapper to make portable.
 * Only the float<->bfloat16 conversions, which no operator performs, are here.
 * Half is the same shape and lives in fp16.cuh; complex diverges the other way --
 * its type has no operators -- and lives in complex.cuh. See
 * docs/architecture.md, section 3.
 *
 * The conversions are __device__-only, like rand.cuh's forwarders: they serve a
 * parallel_for functor's __device__ operator(). Host code reaches the same
 * conversions through bf16.cppm, which wraps them for the host.
 */

#pragma once

// GPUMOD_SELECTED_CUDA / GPUMOD_SELECTED_HIP, and #errors outside a device pass;
// these vendor headers are device-only.
#include "device_guard.h"

#if defined(GPUMOD_SELECTED_CUDA)

#include <cuda_bf16.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by hip_bf16.h) poisons __noinline__ for libc++'s __config, so a
// HIP header reached before <array> makes __config fail to compile. Same
// pre-include the src/hip global module fragments carry. docs/architecture.md,
// section 9.
#include <array>

#include <hip/hip_bf16.h>

#endif

namespace gpumod {

// ========================================================================
// Type -- the same one bf16.cppm exports
// ========================================================================

#if defined(GPUMOD_SELECTED_CUDA)

using gpuBfloat16 = ::__nv_bfloat16;

#else

using gpuBfloat16 = ::__hip_bfloat16;

#endif

/// @brief Convert a float to bfloat16 (round to nearest even)
__device__ __forceinline__ gpuBfloat16 gpuFloat2Bfloat16(const float value) {
  return ::__float2bfloat16(value);
}

/// @brief Widen a bfloat16 value back to float (exact)
__device__ __forceinline__ float gpuBfloat162Float(const gpuBfloat16 value) {
  return ::__bfloat162float(value);
}

} // namespace gpumod
