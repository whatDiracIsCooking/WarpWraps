/**
 * @file fp16.cuh
 * @brief Half type and float conversions for device-compiled TUs
 *
 * The device-compile counterpart to fp16.cppm: wwrHalf and the float<->half
 * conversions. Link wwr.device. Companion to bf16.cuh and complex.h.
 *
 * The type is the SAME one fp16.cppm exports under this name, so a host-allocated
 * buffer and a kernel parameter named here agree, and an extern template declared
 * in a .cppm links against a definition compiled in a .cu.
 *
 * Only the conversions are wrapped. Half carries its arithmetic operators on both
 * backends, so `x + y` on two wwrHalf values is already backend-neutral code
 * naming no vendor symbol -- there is nothing for a wrapper to make portable.
 * Only the float<->half conversions, which no operator performs, are here.
 * bfloat16 is the same shape and lives in bf16.cuh; complex diverges the other
 * way -- its type has no operators -- and lives in complex.h's device section. See
 * docs/architecture.md, section 3.
 *
 * The conversions are __device__-only, like rand.h's device generators: they serve a
 * parallel_for functor's __device__ operator(). Host code reaches the same
 * conversions through fp16.cppm, which wraps them for the host.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass;
// these vendor headers are device-only.
#include "device_guard.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_fp16.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by hip_fp16.h) poisons __noinline__ for libc++'s __config, so a
// HIP header reached before <array> makes __config fail to compile. Same
// pre-include the src/hip global module fragments carry. docs/architecture.md,
// section 9.
#include <array>

#include <hip/hip_fp16.h>

#endif

namespace wwr {

// ========================================================================
// Type -- the same one fp16.cppm exports
// ========================================================================

using wwrHalf = ::__half;

/// @brief Convert a float to half precision (round to nearest even)
__device__ __forceinline__ wwrHalf wwrFloat2Half(const float value) {
  return ::__float2half(value);
}

/// @brief Widen a half-precision value back to float (exact)
__device__ __forceinline__ float wwrHalf2Float(const wwrHalf value) {
  return ::__half2float(value);
}

} // namespace wwr
