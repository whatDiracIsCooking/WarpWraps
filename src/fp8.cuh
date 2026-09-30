/**
 * @file fp8.cuh
 * @brief 8-bit float types and float/double narrowing for device-compiled TUs
 *
 * The device-compile counterpart to fp8.cppm: the wwrFp8* scalar/storage types,
 * the saturation and interpretation enums, and the float/double -> fp8 narrowing
 * conversions. Link wwr.device. Companion to fp16.cuh and bf16.cuh.
 *
 * The types are the SAME ones fp8.cppm exports under these names, so a
 * host-allocated buffer and a kernel parameter named here agree.
 *
 * SCOPED TO THE INTERSECTION, exactly as fp8.cppm: only the OCP e4m3/e5m2 types
 * (with their x2/x4 packed variants) and the __NV_/__HIP_ E4M3/E5M2 enumerators
 * exist on both backends, so those are all this header names -- HIP's fnuz
 * formats and CUDA's e8m0 stay reachable only through the raw modules.
 *
 * Only NARROWING (float/double -> storage) is wrapped, mirroring how fp16.cuh
 * wraps only its conversions. Widening (`static_cast<float>(v)`) and
 * ctor-narrowing (`wwrFp8E4m3{x}`) name no vendor symbol -- once the type alias
 * exists they are already backend-neutral (the operator principle fp16.cuh
 * states) -- and the packed / __half_raw / bf16-raw conversions stay in the raw
 * modules. See docs/architecture.md, section 3.
 *
 * Unlike fp16.cuh, this is the first place the wwr* layer #includes the vendor
 * fp8 header: fp8.cppm reaches its conversions by IMPORTING the raw module's
 * host wrappers, which a device TU cannot do, so a device pass includes the
 * vendor header and forwards to its __host__ __device__ static-inline
 * __nv_cvt_* / __hip_cvt_* conversions directly.
 *
 * The conversions are __device__-only, like rand.h's device generators: they serve a
 * parallel_for functor's __device__ operator(). Host code reaches the same
 * conversions through fp8.cppm, which wraps them for the host.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass;
// these vendor headers are device-only.
#include "device_guard.h"

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

// ========================================================================
// Narrowing conversions (float/double -> fp8 storage)
//
// Forwarding to the vendors' __host__ __device__ static-inline conversions,
// spelled identically apart from the __nv_/__hip_ prefix the switch selects.
// The argument order -- (value, saturation, interpretation) -- is the one both
// vendors declare, matching fp8.cppm's host wrappers.
// ========================================================================

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
