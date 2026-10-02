/**
 * @file bf16.h
 * @brief The bfloat16 type, plus the device float conversions, for the bf16 layer
 *
 * wwrBfloat16, aliased to the vendor's __nv_bfloat16 / __hip_bfloat16 -- the type
 * DIVERGES by backend, so the alias is what makes code above src name it once. It
 * carries two things:
 *   - the type alias (always) -- the one a host-allocated buffer and a kernel
 *     parameter must agree on, and the same one bf16.cppm exports;
 *   - wwrFloat2Bfloat16 / wwrBfloat162Float as __device__ __forceinline__, in a
 *     section gated behind the device-pass macros -- the device half that once
 *     lived in the separate bf16.cuh, so a device .cu includes this one neutral
 *     header. Link wwr.device.
 *
 * Only the conversions are wrapped. bfloat16 carries its arithmetic operators on
 * both backends, so `x + y` on two wwrBfloat16 values is already backend-neutral
 * code naming no vendor symbol. Only the float<->bfloat16 conversions, which no
 * operator performs, are here. Half is the same shape (fp16.h); complex diverges
 * the other way -- its type has no operators -- and lives in complex.h. See
 * docs/architecture.md, section 3.
 *
 * Host code reaches the same conversions through bf16.cppm (import wwr.bf16),
 * which wraps them for the host; the wrappers here are device-only. This is the
 * `.h` + `.cppm` shape for a vendor header with both host and device symbols.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

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

namespace wwr {

// ========================================================================
// Type -- the same one bf16.cppm exports (DIVERGES: __nv_ vs __hip_)
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

using wwrBfloat16 = ::__nv_bfloat16;

#else

using wwrBfloat16 = ::__hip_bfloat16;

#endif

} // namespace wwr

// ========================================================================
// Device float conversions -- present only in a device-compile pass
//
// __device__ __forceinline__, gated behind the compiler's own device-pass macros
// so the type alias above still compiles in a host TU (where __device__ is not a
// keyword, so these must be ABSENT rather than #error). A .cu that #includes
// bf16.h gets them; a host TU gets only the type. This is the device half that
// once lived in the separate bf16.cuh. See docs/architecture.md, section 3.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

// The two conversions, from the one fragment every path shares -- here with the
// device qualifier (bf16.cppm and wwr/bf16.h paste the same list with `inline`).
// See detail/bf16_names.h.
#define WWR_BF16_FN __device__ __forceinline__
#include "detail/bf16_names.h"
#undef WWR_BF16_FN

} // namespace wwr

#endif // device-compile pass
