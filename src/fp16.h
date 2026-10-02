/**
 * @file fp16.h
 * @brief The half type, plus the device float conversions, for the fp16 layer
 *
 * wwrHalf, aliased to the vendor's __half (spelled the same on both backends). It
 * carries two things:
 *   - the type alias (always) -- the one a host-allocated buffer and a kernel
 *     parameter must agree on, and the same one fp16.cppm exports;
 *   - wwrFloat2Half / wwrHalf2Float as __device__ __forceinline__, in a section
 *     gated behind the device-pass macros -- the device half that once lived in
 *     the separate fp16.cuh, so a device .cu includes this one neutral header.
 *     Link wwr.device.
 *
 * Only the conversions are wrapped. Half carries its arithmetic operators on both
 * backends, so `x + y` on two wwrHalf values is already backend-neutral code
 * naming no vendor symbol. Only the float<->half conversions, which no operator
 * performs, are here. bfloat16 is the same shape (bf16.h); complex diverges the
 * other way -- its type has no operators -- and lives in complex.h. See
 * docs/architecture.md, section 3.
 *
 * Host code reaches the same conversions through fp16.cppm (import wwr.fp16),
 * which wraps them for the host; the wrappers here are device-only. This is the
 * `.h` + `.cppm` shape for a vendor header with both host and device symbols.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
#include "selected_backend.h"

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
// Type -- the same one fp16.cppm exports (spelled __half on both backends)
// ========================================================================

using wwrHalf = ::__half;

} // namespace wwr

// ========================================================================
// Device float conversions -- present only in a device-compile pass
//
// __device__ __forceinline__, gated behind the compiler's own device-pass macros
// so the type alias above still compiles in a host TU (where __device__ is not a
// keyword, so these must be ABSENT rather than #error). A .cu that #includes
// fp16.h gets them; a host TU gets only the type. This is the device half that
// once lived in the separate fp16.cuh. See docs/architecture.md, section 3.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

// The two conversions, from the one fragment every path shares -- here with the
// device qualifier (fp16.cppm and wwr/fp16.h paste the same list with `inline`).
// See detail/fp16_names.h.
#define WWR_FP16_FN __device__ __forceinline__
#include "detail/fp16_names.h"
#undef WWR_FP16_FN

} // namespace wwr

#endif // device-compile pass
