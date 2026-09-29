/**
 * @file complex.cuh
 * @brief Complex types and portable arithmetic for device-compiled TUs
 *
 * The device-compile counterpart to complex.cppm: wwrFloatComplex /
 * wwrDoubleComplex / wwrComplex, make_wwr*Complex, the wwrC* arithmetic and
 * accessors, and the wwrComplexFloatToDouble / wwrComplexDoubleToFloat precision
 * conversions. Link wwr.device. Companion to fp16.cuh and bf16.cuh.
 *
 * The types are the SAME ones complex.cppm exports under these names, so a
 * host-allocated buffer and a kernel parameter named here agree, and an extern
 * template declared in a .cppm links against a definition compiled in a .cu.
 *
 * Construction and arithmetic go through wwr* functions rather than operators
 * because cuComplex is an operator-less float2 aggregate where hipComplex is a
 * class -- so `a * b` and brace-initialisation are not portable, and the
 * vendors' C-style functions are the only spelling that exists on both. Half
 * and bfloat16 diverge the other way and live in fp16.cuh / bf16.cuh. See
 * docs/architecture.md, section 3.
 *
 * The wrappers are __device__ __forceinline__, like fp16.cuh's conversions:
 * they serve a parallel_for functor's __device__ operator(). Host code reaches
 * the same construction and arithmetic through complex.cppm, which wraps them
 * for the host -- this header is not on the host path.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass;
// these vendor headers are device-only.
#include "device_guard.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuComplex.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the HIP vendor headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include the src/hip global module fragments carry.
// docs/architecture.md, section 9.
#include <array>

#include <hip/hip_complex.h>

#endif

namespace wwr {

// ========================================================================
// Types -- the same ones complex.cppm exports
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

using wwrFloatComplex = ::cuFloatComplex;
using wwrDoubleComplex = ::cuDoubleComplex;
using wwrComplex = ::cuComplex;

#else

using wwrFloatComplex = ::hipFloatComplex;
using wwrDoubleComplex = ::hipDoubleComplex;
using wwrComplex = ::hipComplex;

#endif

// ========================================================================
// Construction
//
// __device__ __forceinline__, like fp16.cuh: these serve a parallel_for
// functor's __device__ operator(). A host TU reaches the same construction
// through complex.cppm, not this header.
// ========================================================================

/// @brief Build a single-precision complex value from its two components
__device__ __forceinline__ wwrFloatComplex make_wwrFloatComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuFloatComplex(re, im);
#else
  return ::make_hipFloatComplex(re, im);
#endif
}

/// @brief Build a double-precision complex value from its two components
__device__ __forceinline__ wwrDoubleComplex make_wwrDoubleComplex(const double re,
                                                                  const double im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuDoubleComplex(re, im);
#else
  return ::make_hipDoubleComplex(re, im);
#endif
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_wwrFloatComplex -- wwrComplex is wwrFloatComplex)
__device__ __forceinline__ wwrComplex make_wwrComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuComplex(re, im);
#else
  return ::make_hipComplex(re, im);
#endif
}

// ========================================================================
// Arithmetic and accessors
//
// Through wwr* functions rather than operators, for the same reason
// construction is (docs/architecture.md §3): cuFloatComplex is a plain float2
// with no arithmetic operators, so `a * b` compiles under HIP -- whose
// hipComplex is a class that defines them -- and fails under CUDA with no
// operator match. The vendors' C-style functions exist on both and are the
// portable spelling; the wwr* names carry the divergent cu*/hip* spellings.
//
// __device__ __forceinline__ like construction above. The arguments are taken
// by value, as the vendors declare them -- wwrFloatComplex is 8 bytes,
// wwrDoubleComplex 16.
// ========================================================================

/// @brief Real part of a single-precision complex value
__device__ __forceinline__ float wwrCrealf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCrealf(z);
#else
  return ::hipCrealf(z);
#endif
}

/// @brief Imaginary part of a single-precision complex value
__device__ __forceinline__ float wwrCimagf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimagf(z);
#else
  return ::hipCimagf(z);
#endif
}

/// @brief Real part of a double-precision complex value
__device__ __forceinline__ double wwrCreal(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCreal(z);
#else
  return ::hipCreal(z);
#endif
}

/// @brief Imaginary part of a double-precision complex value
__device__ __forceinline__ double wwrCimag(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimag(z);
#else
  return ::hipCimag(z);
#endif
}

/// @brief Magnitude (absolute value) of a single-precision complex value
__device__ __forceinline__ float wwrCabsf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabsf(z);
#else
  return ::hipCabsf(z);
#endif
}

/// @brief Magnitude (absolute value) of a double-precision complex value
__device__ __forceinline__ double wwrCabs(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabs(z);
#else
  return ::hipCabs(z);
#endif
}

/// @brief Complex conjugate of a single-precision complex value
__device__ __forceinline__ wwrFloatComplex wwrConjf(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConjf(z);
#else
  return ::hipConjf(z);
#endif
}

/// @brief Complex conjugate of a double-precision complex value
__device__ __forceinline__ wwrDoubleComplex wwrConj(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConj(z);
#else
  return ::hipConj(z);
#endif
}

/// @brief Sum of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCaddf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCaddf(a, b);
#else
  return ::hipCaddf(a, b);
#endif
}

/// @brief Difference of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCsubf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsubf(a, b);
#else
  return ::hipCsubf(a, b);
#endif
}

/// @brief Product of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCmulf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmulf(a, b);
#else
  return ::hipCmulf(a, b);
#endif
}

/// @brief Quotient of two single-precision complex values
__device__ __forceinline__ wwrFloatComplex wwrCdivf(const wwrFloatComplex a,
                                                    const wwrFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCdivf(a, b);
#else
  return ::hipCdivf(a, b);
#endif
}

/// @brief Sum of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCadd(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCadd(a, b);
#else
  return ::hipCadd(a, b);
#endif
}

/// @brief Difference of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCsub(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsub(a, b);
#else
  return ::hipCsub(a, b);
#endif
}

/// @brief Product of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCmul(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmul(a, b);
#else
  return ::hipCmul(a, b);
#endif
}

/// @brief Quotient of two double-precision complex values
__device__ __forceinline__ wwrDoubleComplex wwrCdiv(const wwrDoubleComplex a,
                                                    const wwrDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCdiv(a, b);
#else
  return ::hipCdiv(a, b);
#endif
}

// ========================================================================
// Precision conversion
//
// The other portable functions the vendors name with the bare Complex token
// (alongside make_wwrComplex above): widen a single-precision value to double,
// or narrow it back. HIP's Complex-named extras -- hipCsqabs*, hipCfma* -- have
// no cuComplex.h counterpart, so they are not part of the wwr* surface.
// ========================================================================

/// @brief Widen a single-precision complex value to double precision
__device__ __forceinline__ wwrDoubleComplex wwrComplexFloatToDouble(const wwrFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexFloatToDouble(z);
#else
  return ::hipComplexFloatToDouble(z);
#endif
}

/// @brief Narrow a double-precision complex value to single precision
__device__ __forceinline__ wwrFloatComplex wwrComplexDoubleToFloat(const wwrDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexDoubleToFloat(z);
#else
  return ::hipComplexDoubleToFloat(z);
#endif
}

} // namespace wwr
