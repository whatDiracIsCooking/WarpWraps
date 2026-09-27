/**
 * @file complex.cuh
 * @brief Complex types and portable arithmetic for device-compiled TUs
 *
 * The device-compile counterpart to complex.cppm: gpuFloatComplex /
 * gpuDoubleComplex / gpuComplex, make_gpu*Complex, the gpuC* arithmetic and
 * accessors, and the gpuComplexFloatToDouble / gpuComplexDoubleToFloat precision
 * conversions. Link wwr.device. Companion to fp16.cuh and bf16.cuh.
 *
 * The types are the SAME ones complex.cppm exports under these names, so a
 * host-allocated buffer and a kernel parameter named here agree, and an extern
 * template declared in a .cppm links against a definition compiled in a .cu.
 *
 * Construction and arithmetic go through gpu* functions rather than operators
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

using gpuFloatComplex = ::cuFloatComplex;
using gpuDoubleComplex = ::cuDoubleComplex;
using gpuComplex = ::cuComplex;

#else

using gpuFloatComplex = ::hipFloatComplex;
using gpuDoubleComplex = ::hipDoubleComplex;
using gpuComplex = ::hipComplex;

#endif

// ========================================================================
// Construction
//
// __device__ __forceinline__, like fp16.cuh: these serve a parallel_for
// functor's __device__ operator(). A host TU reaches the same construction
// through complex.cppm, not this header.
// ========================================================================

/// @brief Build a single-precision complex value from its two components
__device__ __forceinline__ gpuFloatComplex make_gpuFloatComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuFloatComplex(re, im);
#else
  return ::make_hipFloatComplex(re, im);
#endif
}

/// @brief Build a double-precision complex value from its two components
__device__ __forceinline__ gpuDoubleComplex make_gpuDoubleComplex(const double re,
                                                                  const double im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuDoubleComplex(re, im);
#else
  return ::make_hipDoubleComplex(re, im);
#endif
}

/// @brief Build a single-precision complex value (the vendors' make_*Complex
///        alias for make_gpuFloatComplex -- gpuComplex is gpuFloatComplex)
__device__ __forceinline__ gpuComplex make_gpuComplex(const float re, const float im) {
#if defined(WWR_SELECTED_CUDA)
  return ::make_cuComplex(re, im);
#else
  return ::make_hipComplex(re, im);
#endif
}

// ========================================================================
// Arithmetic and accessors
//
// Through gpu* functions rather than operators, for the same reason
// construction is (docs/architecture.md §3): cuFloatComplex is a plain float2
// with no arithmetic operators, so `a * b` compiles under HIP -- whose
// hipComplex is a class that defines them -- and fails under CUDA with no
// operator match. The vendors' C-style functions exist on both and are the
// portable spelling; the gpu* names carry the divergent cu*/hip* spellings.
//
// __device__ __forceinline__ like construction above. The arguments are taken
// by value, as the vendors declare them -- gpuFloatComplex is 8 bytes,
// gpuDoubleComplex 16.
// ========================================================================

/// @brief Real part of a single-precision complex value
__device__ __forceinline__ float gpuCrealf(const gpuFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCrealf(z);
#else
  return ::hipCrealf(z);
#endif
}

/// @brief Imaginary part of a single-precision complex value
__device__ __forceinline__ float gpuCimagf(const gpuFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimagf(z);
#else
  return ::hipCimagf(z);
#endif
}

/// @brief Real part of a double-precision complex value
__device__ __forceinline__ double gpuCreal(const gpuDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCreal(z);
#else
  return ::hipCreal(z);
#endif
}

/// @brief Imaginary part of a double-precision complex value
__device__ __forceinline__ double gpuCimag(const gpuDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCimag(z);
#else
  return ::hipCimag(z);
#endif
}

/// @brief Magnitude (absolute value) of a single-precision complex value
__device__ __forceinline__ float gpuCabsf(const gpuFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabsf(z);
#else
  return ::hipCabsf(z);
#endif
}

/// @brief Magnitude (absolute value) of a double-precision complex value
__device__ __forceinline__ double gpuCabs(const gpuDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCabs(z);
#else
  return ::hipCabs(z);
#endif
}

/// @brief Complex conjugate of a single-precision complex value
__device__ __forceinline__ gpuFloatComplex gpuConjf(const gpuFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConjf(z);
#else
  return ::hipConjf(z);
#endif
}

/// @brief Complex conjugate of a double-precision complex value
__device__ __forceinline__ gpuDoubleComplex gpuConj(const gpuDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuConj(z);
#else
  return ::hipConj(z);
#endif
}

/// @brief Sum of two single-precision complex values
__device__ __forceinline__ gpuFloatComplex gpuCaddf(const gpuFloatComplex a,
                                                    const gpuFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCaddf(a, b);
#else
  return ::hipCaddf(a, b);
#endif
}

/// @brief Difference of two single-precision complex values
__device__ __forceinline__ gpuFloatComplex gpuCsubf(const gpuFloatComplex a,
                                                    const gpuFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsubf(a, b);
#else
  return ::hipCsubf(a, b);
#endif
}

/// @brief Product of two single-precision complex values
__device__ __forceinline__ gpuFloatComplex gpuCmulf(const gpuFloatComplex a,
                                                    const gpuFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmulf(a, b);
#else
  return ::hipCmulf(a, b);
#endif
}

/// @brief Quotient of two single-precision complex values
__device__ __forceinline__ gpuFloatComplex gpuCdivf(const gpuFloatComplex a,
                                                    const gpuFloatComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCdivf(a, b);
#else
  return ::hipCdivf(a, b);
#endif
}

/// @brief Sum of two double-precision complex values
__device__ __forceinline__ gpuDoubleComplex gpuCadd(const gpuDoubleComplex a,
                                                    const gpuDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCadd(a, b);
#else
  return ::hipCadd(a, b);
#endif
}

/// @brief Difference of two double-precision complex values
__device__ __forceinline__ gpuDoubleComplex gpuCsub(const gpuDoubleComplex a,
                                                    const gpuDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCsub(a, b);
#else
  return ::hipCsub(a, b);
#endif
}

/// @brief Product of two double-precision complex values
__device__ __forceinline__ gpuDoubleComplex gpuCmul(const gpuDoubleComplex a,
                                                    const gpuDoubleComplex b) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuCmul(a, b);
#else
  return ::hipCmul(a, b);
#endif
}

/// @brief Quotient of two double-precision complex values
__device__ __forceinline__ gpuDoubleComplex gpuCdiv(const gpuDoubleComplex a,
                                                    const gpuDoubleComplex b) {
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
// (alongside make_gpuComplex above): widen a single-precision value to double,
// or narrow it back. HIP's Complex-named extras -- hipCsqabs*, hipCfma* -- have
// no cuComplex.h counterpart, so they are not part of the gpu* surface.
// ========================================================================

/// @brief Widen a single-precision complex value to double precision
__device__ __forceinline__ gpuDoubleComplex gpuComplexFloatToDouble(const gpuFloatComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexFloatToDouble(z);
#else
  return ::hipComplexFloatToDouble(z);
#endif
}

/// @brief Narrow a double-precision complex value to single precision
__device__ __forceinline__ gpuFloatComplex gpuComplexDoubleToFloat(const gpuDoubleComplex z) {
#if defined(WWR_SELECTED_CUDA)
  return ::cuComplexDoubleToFloat(z);
#else
  return ::hipComplexDoubleToFloat(z);
#endif
}

} // namespace wwr
