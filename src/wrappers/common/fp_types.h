/**
 * @file fp_types.h
 * @brief Floating-point type concepts and real/complex/half maps for GPU ops
 *
 * The C++20 concepts (real_fp, complex_fp, usual_fp, half_fp, usual_and_half_fp)
 * and the type maps (RealToComplexType, ComplexToRealType, HalfToFloatType) the
 * four extensions are written against. They constrain template parameters to the
 * backend-neutral wwr* floating-point types, so they match cuComplex /
 * __nv_bfloat16 on a CUDA build and hipComplex / __hip_bfloat16 on a HIP build.
 *
 * Unlike complex.h / fp16.h / bf16.h, this header carries NO device-pass-gated
 * section: concepts and alias templates are pure compile-time constructs with no
 * __device__ bodies, so the one definition serves a host TU and a device .cu
 * alike. It is what the module partition fp_types.cppm re-exports (import
 * wwr.wrappers.common), and the same header a device .cu includes directly --
 * a device TU cannot import a module, so the concepts reach kernel code this way.
 *
 * The wwr* types the concepts name come from complex.h / fp16.h / bf16.h, the
 * device-safe type headers; <type_traits> supplies is_same_v / conditional_t.
 * See docs/architecture.md, section 3.
 */

#pragma once

// <type_traits> supplies the concepts' is_same_v / conditional_t. <array> and
// <algorithm> are load-bearing pre-includes for the HIP backend: this header
// pulls bf16.h then fp16.h into one TU -- the same order amd_hip_fp8.h uses --
// so amd_hip_bf16.h's device_library_decls.h poisons __local before
// amd_hip_fp16.h does the TU's first <algorithm>, and host_defines.h poisons
// __noinline__ for libc++'s __config before <array>. Seeing both libc++ headers
// first, while the macros are still clean, sidesteps both. They must stay before
// the vendor type headers. docs/architecture.md, sections 9 and 10.
#include <algorithm>
#include <array>
#include <type_traits>

// The backend-neutral wwr* floating-point types the concepts below name. These
// are the device-safe `.h` faces (not the modules), so this header is includable
// from a device .cu as well as from fp_types.cppm's global module fragment.
#include "bf16.h"
#include "complex.h"
#include "fp16.h"

namespace wwr {

// ========================================================================
// Floating-Point Type Concepts
// ========================================================================

/**
 * @brief Concept constraining type T to real floating-point types
 *
 * Constrains T to be either float or double, which are the real-valued
 * floating-point types supported by GPU BLAS operations.
 */
template<typename T>
concept real_fp = std::is_same_v<T, float> || std::is_same_v<T, double>;

/**
 * @brief Concept constraining type T to GPU complex floating-point types
 *
 * Constrains T to be either wwrFloatComplex or wwrDoubleComplex, which are
 * the complex-valued floating-point types supported by GPU BLAS operations.
 */
template<typename T>
concept complex_fp = std::is_same_v<T, wwrFloatComplex> || std::is_same_v<T, wwrDoubleComplex>;

/**
 * @brief Concept constraining type T to usual GPU BLAS floating-point types
 *
 * Constrains T to be any of the usual floating-point types supported by GPU BLAS operations,
 * including both real (float, double) and complex (wwrFloatComplex, wwrDoubleComplex) types.
 */
template<typename T>
concept usual_fp = real_fp<T> || complex_fp<T>;

/**
 * @brief Concept constraining type T to GPU half-precision floating-point types
 *
 * Constrains T to be either wwrHalf or wwrBfloat16, which are the
 * half-precision floating-point types supported by GPU BLAS operations.
 */
template<typename T>
concept half_fp = std::is_same_v<T, wwrHalf> || std::is_same_v<T, wwrBfloat16>;

/**
 * @brief Concept constraining type T to usual or half-precision GPU BLAS floating-point types
 *
 * Constrains T to be any floating-point type supported by GPU BLAS operations,
 * including usual types (float, double, wwrFloatComplex, wwrDoubleComplex) and
 * half-precision types (wwrHalf, wwrBfloat16).
 */
template<typename T>
concept usual_and_half_fp = usual_fp<T> || half_fp<T>;

// ========================================================================
// Real <-> Complex Type Mappings
// ========================================================================

/**
 * @brief Maps a real_fp type to its corresponding GPU complex type.
 *   float  -> wwrFloatComplex
 *   double -> wwrDoubleComplex
 */
template<real_fp T>
using RealToComplexType =
    std::conditional_t<std::is_same_v<T, float>, wwrFloatComplex, wwrDoubleComplex>;

/**
 * @brief Maps a usual_fp type to its underlying real scalar type.
 *   float           -> float
 *   double          -> double
 *   wwrFloatComplex  -> float
 *   wwrDoubleComplex -> double
 */
template<usual_fp T>
using ComplexToRealType =
    std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, wwrFloatComplex>, float,
                       double>;

// ========================================================================
// Half-Precision Type Mappings
// ========================================================================

/**
 * @brief Maps a half_fp type to its corresponding single-precision type.
 *   wwrHalf     -> float
 *   wwrBfloat16 -> float
 */
template<half_fp T>
using HalfToFloatType = float;

} // namespace wwr
