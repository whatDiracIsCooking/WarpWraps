/**
 * @file fp_types.cppm
 * @brief Floating-point type concepts for GPU operations
 *
 * This module provides C++20 concepts for constraining template parameters
 * to floating-point types commonly used in GPU BLAS/solver libraries. The
 * complex and half-precision types are the backend-neutral gpu* aliases from
 * src, so the concepts match cuComplex/__nv_bfloat16 on a CUDA build and
 * hipComplex/__hip_bfloat16 on a HIP build.
 *
 * Usage:
 *   import gpumod.wrappers.common;
 *   using namespace gpumod;
 */

export module gpumod.wrappers.common:fp_types;

import std;
import gpumod.complex;
import gpumod.fp16;
import gpumod.bf16;

export namespace gpumod {

// ========================================================================
// GPU Complex Types
// ========================================================================

using gpumod::gpuComplex;
using gpumod::gpuDoubleComplex;
using gpumod::gpuFloatComplex;

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
 * Constrains T to be either gpuFloatComplex or gpuDoubleComplex, which are
 * the complex-valued floating-point types supported by GPU BLAS operations.
 */
template<typename T>
concept complex_fp = std::is_same_v<T, gpuFloatComplex> || std::is_same_v<T, gpuDoubleComplex>;

/**
 * @brief Concept constraining type T to usual GPU BLAS floating-point types
 *
 * Constrains T to be any of the usual floating-point types supported by GPU BLAS operations,
 * including both real (float, double) and complex (gpuFloatComplex, gpuDoubleComplex) types.
 */
template<typename T>
concept usual_fp = real_fp<T> || complex_fp<T>;

/**
 * @brief Concept constraining type T to GPU half-precision floating-point types
 *
 * Constrains T to be either gpuHalf or gpuBfloat16, which are the
 * half-precision floating-point types supported by GPU BLAS operations.
 */
template<typename T>
concept half_fp = std::is_same_v<T, gpuHalf> || std::is_same_v<T, gpuBfloat16>;

/**
 * @brief Concept constraining type T to usual or half-precision GPU BLAS floating-point types
 *
 * Constrains T to be any floating-point type supported by GPU BLAS operations,
 * including usual types (float, double, gpuFloatComplex, gpuDoubleComplex) and
 * half-precision types (gpuHalf, gpuBfloat16).
 */
template<typename T>
concept usual_and_half_fp = usual_fp<T> || half_fp<T>;

// ========================================================================
// Real <-> Complex Type Mappings
// ========================================================================

/**
 * @brief Maps a real_fp type to its corresponding GPU complex type.
 *   float  -> gpuFloatComplex
 *   double -> gpuDoubleComplex
 */
template<real_fp T>
using RealToComplexType =
    std::conditional_t<std::is_same_v<T, float>, gpuFloatComplex, gpuDoubleComplex>;

/**
 * @brief Maps a usual_fp type to its underlying real scalar type.
 *   float           -> float
 *   double          -> double
 *   gpuFloatComplex  -> float
 *   gpuDoubleComplex -> double
 */
template<usual_fp T>
using ComplexToRealType =
    std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, gpuFloatComplex>, float,
                       double>;

// ========================================================================
// Half-Precision Type Mappings
// ========================================================================

/**
 * @brief Maps a half_fp type to its corresponding single-precision type.
 *   gpuHalf     -> float
 *   gpuBfloat16 -> float
 */
template<half_fp T>
using HalfToFloatType = float;

} // namespace gpumod
