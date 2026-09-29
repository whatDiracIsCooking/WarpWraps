/**
 * @file bf16.cppm
 * @brief Backend-neutral bfloat16 type: wwrBfloat16 for __nv_bfloat16 / __hip_bfloat16
 *
 * See backend.h.
 *
 * Usage:
 *   import wwr.bf16;
 */

module;

#include "backend.h"

// The float<->bfloat16 conversions are static-inline in the vendor header (see
// docs/architecture.md section 12), so the vendor module cannot export them and
// an import cannot reach them -- the header itself must be in the GMF, where its
// external linkage is preserved (section 14). Host-only: a device TU reaches the
// same conversions through fp16.cuh instead.
#if defined(WWR_GPU_BACKEND_CUDA)
#include <cuda_bf16.h>
#else
// <array> before any HIP header. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_bf16.h>
#endif

export module wwr.bf16;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_bf16;
#else
import wwr.hip.hip_bf16;
#endif

export namespace wwr {

WWR_TYPE(wwrBfloat16, __nv_bfloat16, __hip_bfloat16)

/// @brief Convert a float to bfloat16 (round to nearest even)
inline wwrBfloat16 wwrFloat2Bfloat16(const float value) {
  return ::__float2bfloat16(value);
}

/// @brief Widen a bfloat16 value back to float (exact)
inline float wwrBfloat162Float(const wwrBfloat16 value) {
  return ::__bfloat162float(value);
}

} // namespace wwr
