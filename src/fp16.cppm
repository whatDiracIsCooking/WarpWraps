/**
 * @file fp16.cppm
 * @brief Backend-neutral half-precision type: gpuHalf for __half
 *
 * Both backends spell the type __half (cuda_fp16.h / hip_fp16.h); the alias
 * exists so code above src names every backend type the same way.
 * See gpu_backend.h.
 *
 * Usage:
 *   import gpumod.fp16;
 */

module;

#include "gpu_backend.h"

// The float<->half conversions are static-inline in the vendor header (see
// docs/architecture.md section 12), so the vendor module cannot export them and
// an import cannot reach them -- the header itself must be in the GMF, where its
// external linkage is preserved (section 14). Host-only: a device TU reaches the
// same conversions through fp16.cuh instead.
#if defined(GPUMOD_GPU_BACKEND_CUDA)
#include <cuda_fp16.h>
#else
// <array> before any HIP header. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_fp16.h>
#endif

export module gpumod.fp16;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_fp16;
#else
import gpumod.hip.hip_fp16;
#endif

export namespace gpumod {

GPUMOD_TYPE(gpuHalf, __half, __half)

/// @brief Convert a float to half precision (round to nearest even)
inline gpuHalf gpuFloat2Half(const float value) {
  return ::__float2half(value);
}

/// @brief Widen a half-precision value back to float (exact)
inline float gpuHalf2Float(const gpuHalf value) {
  return ::__half2float(value);
}

} // namespace gpumod
