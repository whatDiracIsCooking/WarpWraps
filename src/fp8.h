/**
 * @file fp8.h
 * @brief The 8-bit float types, plus device narrowing, for the fp8 layer
 *
 * The wwrFp8* scalar/storage types, the saturation and interpretation enums, and
 * the float/double -> fp8 narrowing conversions. It carries two things:
 *   - the type aliases and enum constants (always) -- the ones a host-allocated
 *     buffer and a kernel parameter must agree on, and the same ones fp8.cppm
 *     exports;
 *   - wwrFloat2Fp8 / wwrDouble2Fp8 as __device__ __forceinline__, in a section
 *     gated behind the device-pass macros -- the device half that once lived in
 *     the separate fp8.cuh, so a device .cu includes this one neutral header.
 *     Link wwr.device.
 *
 * SCOPED TO THE INTERSECTION, exactly as fp8.cppm: only the OCP e4m3/e5m2 types
 * (with their x2/x4 packed variants) and the __NV_/__HIP_ E4M3/E5M2 enumerators
 * exist on both backends, so those are all this header names -- HIP's fnuz
 * formats and CUDA's e8m0 stay reachable only through the raw modules.
 *
 * Only NARROWING (float/double -> storage) is wrapped. Widening
 * (`static_cast<float>(v)`) and ctor-narrowing (`wwrFp8E4m3{x}`) name no vendor
 * symbol -- once the type alias exists they are already backend-neutral (the
 * operator principle fp16.h states) -- and the packed / __half_raw / bf16-raw
 * conversions stay in the raw modules. See docs/architecture.md, section 3.
 *
 * The host reaches the same conversions through fp8.cppm (import wwr.fp8), which
 * unlike this header IMPORTS the raw module's host wrappers for the static-inline
 * __nv_cvt_* / __hip_cvt_* conversions (docs/architecture.md section 12); a device
 * pass cannot import, so the device wrappers here #include the vendor header and
 * forward to those static-inline conversions directly. This is the `.h` + `.cppm`
 * shape for a vendor header with both host and device symbols.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_fp8.h>

#else

// Load-bearing, and both must stay before the HIP header, for two different
// reasons the raw wwr.hip.hip_fp8 module carries too: host_defines.h (pulled in
// transitively) poisons __noinline__ for libc++'s __config, so a HIP header
// reached before <array> makes __config fail to compile (docs/architecture.md
// section 9); and device_library_decls.h poisons __local for libc++'s
// <algorithm>, so that must be seen first as well (section 10).
#include <algorithm>
#include <array>

#include <hip/hip_fp8.h>

#endif

namespace wwr {

// ========================================================================
// Types and enums -- the same ones fp8.cppm exports
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

using wwrFp8Storage = ::__nv_fp8_storage_t;
using wwrFp8x2Storage = ::__nv_fp8x2_storage_t;
using wwrFp8x4Storage = ::__nv_fp8x4_storage_t;

using wwrSaturation = ::__nv_saturation_t;
inline constexpr wwrSaturation wwrNosat = ::__NV_NOSAT;
inline constexpr wwrSaturation wwrSatfinite = ::__NV_SATFINITE;

using wwrFp8Interpretation = ::__nv_fp8_interpretation_t;
inline constexpr wwrFp8Interpretation wwrE4m3 = ::__NV_E4M3;
inline constexpr wwrFp8Interpretation wwrE5m2 = ::__NV_E5M2;

using wwrFp8E4m3 = ::__nv_fp8_e4m3;
using wwrFp8x2E4m3 = ::__nv_fp8x2_e4m3;
using wwrFp8x4E4m3 = ::__nv_fp8x4_e4m3;

using wwrFp8E5m2 = ::__nv_fp8_e5m2;
using wwrFp8x2E5m2 = ::__nv_fp8x2_e5m2;
using wwrFp8x4E5m2 = ::__nv_fp8x4_e5m2;

#else

using wwrFp8Storage = ::__hip_fp8_storage_t;
using wwrFp8x2Storage = ::__hip_fp8x2_storage_t;
using wwrFp8x4Storage = ::__hip_fp8x4_storage_t;

using wwrSaturation = ::__hip_saturation_t;
inline constexpr wwrSaturation wwrNosat = ::__HIP_NOSAT;
inline constexpr wwrSaturation wwrSatfinite = ::__HIP_SATFINITE;

using wwrFp8Interpretation = ::__hip_fp8_interpretation_t;
inline constexpr wwrFp8Interpretation wwrE4m3 = ::__HIP_E4M3;
inline constexpr wwrFp8Interpretation wwrE5m2 = ::__HIP_E5M2;

using wwrFp8E4m3 = ::__hip_fp8_e4m3;
using wwrFp8x2E4m3 = ::__hip_fp8x2_e4m3;
using wwrFp8x4E4m3 = ::__hip_fp8x4_e4m3;

using wwrFp8E5m2 = ::__hip_fp8_e5m2;
using wwrFp8x2E5m2 = ::__hip_fp8x2_e5m2;
using wwrFp8x4E5m2 = ::__hip_fp8x4_e5m2;

#endif

} // namespace wwr

// ========================================================================
// Device narrowing conversions (float/double -> fp8 storage) -- present only in
// a device-compile pass
//
// __device__ __forceinline__, gated behind the compiler's own device-pass macros
// so the types above still compile in a host TU (where __device__ is not a
// keyword, so these must be ABSENT rather than #error). Forwarding to the
// vendors' __host__ __device__ static-inline conversions, spelled identically
// apart from the __nv_/__hip_ prefix the switch selects. The argument order --
// (value, saturation, interpretation) -- is the one both vendors declare,
// matching fp8.cppm's host wrappers. This is the device half that once lived in
// the separate fp8.cuh. See docs/architecture.md, section 3.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

/// @brief Narrow a float to an fp8 value in the given format (round to nearest)
__device__ __forceinline__ wwrFp8Storage wwrFloat2Fp8(const float value,
                                                       const wwrSaturation saturate,
                                                       const wwrFp8Interpretation interpretation) {
#if defined(WWR_SELECTED_CUDA)
  return ::__nv_cvt_float_to_fp8(value, saturate, interpretation);
#else
  return ::__hip_cvt_float_to_fp8(value, saturate, interpretation);
#endif
}

/// @brief Narrow a double to an fp8 value in the given format (round to nearest)
__device__ __forceinline__ wwrFp8Storage wwrDouble2Fp8(const double value,
                                                        const wwrSaturation saturate,
                                                        const wwrFp8Interpretation interpretation) {
#if defined(WWR_SELECTED_CUDA)
  return ::__nv_cvt_double_to_fp8(value, saturate, interpretation);
#else
  return ::__hip_cvt_double_to_fp8(value, saturate, interpretation);
#endif
}

} // namespace wwr

#endif // device-compile pass
