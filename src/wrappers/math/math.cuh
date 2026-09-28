/**
 * @file math.cuh
 * @brief Element-wise transcendental math, templated on float/double, for device TUs
 *
 * Type-generic wrappers over the vendor device math intrinsics: one template per
 * function (exp, log, sqrt, sin, ...) that dispatches on the element type to the
 * single- or double-precision entry point (::expf vs ::exp, and so on). Written
 * once, they let a parallel_for functor's __device__ operator() call
 * `wwr::exp(x)` without spelling the `f`-suffixed name or repeating the
 * float/double branch at every call site. Link wwr.device.
 *
 * __device__ __forceinline__, like fp16.cuh's conversions and complex.cuh's
 * arithmetic: these serve a kernel body and carry no host counterpart -- host
 * code reaches the same functions through <cmath>. Only float and double are
 * supported; any other T is a compile error, so a stray half or complex argument
 * is caught here rather than silently narrowing.
 *
 * The intrinsics themselves live in the global namespace and are declared by the
 * vendor runtime header (which also defines __device__/__forceinline__); both
 * backends spell them the same, so the only per-backend branch is which runtime
 * header to pull. Companion to complex.cuh, fp16.cuh and bf16.cuh; see
 * docs/architecture.md, section 3.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and #errors outside a device pass:
// the intrinsics below exist only in a device compile.
#include "device_guard.h"

// std::is_same_v for the compile-time float/double branch. Kept before the
// vendor runtime header below: it pulls libc++'s __config, which a HIP header
// reached first would poison (host_defines.h clobbers __noinline__ -- the same
// ordering fp16.cuh documents).
#include <type_traits>

// The math intrinsics (::expf, ::exp, ...) and __device__/__forceinline__.
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

namespace wwr {

// One template per unary function: the f-suffixed intrinsic for float, the
// bare one for double, chosen at compile time. static_assert rejects any other
// element type up front.
#define WWR_MATH_UNARY(fn, ffn, dfn)                                                                \
  template <typename T> __device__ __forceinline__ T fn(const T x) {                                \
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                            \
                  "wwr::" #fn " supports only float and double");                                   \
    if constexpr (std::is_same_v<T, float>) {                                                       \
      return ::ffn(x);                                                                              \
    } else {                                                                                        \
      return ::dfn(x);                                                                              \
    }                                                                                               \
  }

// Exponential / logarithmic. expm1 and log1p are not redundant with exp/log:
// they keep precision for arguments near zero, which is the whole reason a
// kernel would reach for them over `exp(x) - 1`.
WWR_MATH_UNARY(exp, expf, exp)
WWR_MATH_UNARY(exp2, exp2f, exp2)
WWR_MATH_UNARY(expm1, expm1f, expm1)
WWR_MATH_UNARY(log, logf, log)
WWR_MATH_UNARY(log2, log2f, log2)
WWR_MATH_UNARY(log10, log10f, log10)
WWR_MATH_UNARY(log1p, log1pf, log1p)

// Powers / roots.
WWR_MATH_UNARY(sqrt, sqrtf, sqrt)
WWR_MATH_UNARY(rsqrt, rsqrtf, rsqrt)

// Trigonometric, inverse trigonometric, hyperbolic. atan has a two-argument
// sibling (atan2) in the binary block below.
WWR_MATH_UNARY(sin, sinf, sin)
WWR_MATH_UNARY(cos, cosf, cos)
WWR_MATH_UNARY(tan, tanf, tan)
WWR_MATH_UNARY(asin, asinf, asin)
WWR_MATH_UNARY(acos, acosf, acos)
WWR_MATH_UNARY(atan, atanf, atan)
WWR_MATH_UNARY(sinh, sinhf, sinh)
WWR_MATH_UNARY(cosh, coshf, cosh)
WWR_MATH_UNARY(tanh, tanhf, tanh)

// Rounding to integral values (still returning the floating type).
WWR_MATH_UNARY(floor, floorf, floor)
WWR_MATH_UNARY(ceil, ceilf, ceil)
WWR_MATH_UNARY(trunc, truncf, trunc)
WWR_MATH_UNARY(round, roundf, round)
WWR_MATH_UNARY(rint, rintf, rint)

WWR_MATH_UNARY(fabs, fabsf, fabs)

#undef WWR_MATH_UNARY

// The two-argument form, same dispatch shape.
#define WWR_MATH_BINARY(fn, ffn, dfn)                                                               \
  template <typename T> __device__ __forceinline__ T fn(const T x, const T y) {                     \
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                            \
                  "wwr::" #fn " supports only float and double");                                   \
    if constexpr (std::is_same_v<T, float>) {                                                       \
      return ::ffn(x, y);                                                                           \
    } else {                                                                                        \
      return ::dfn(x, y);                                                                           \
    }                                                                                               \
  }

WWR_MATH_BINARY(pow, powf, pow)
WWR_MATH_BINARY(atan2, atan2f, atan2)
WWR_MATH_BINARY(hypot, hypotf, hypot)
WWR_MATH_BINARY(fmod, fmodf, fmod)
WWR_MATH_BINARY(copysign, copysignf, copysign)
// fmin/fmax are the ordered-min/max intrinsics, not `x < y ? x : y`: they return
// the non-NaN operand when one is NaN, which a bare comparison does not.
WWR_MATH_BINARY(fmin, fminf, fmin)
WWR_MATH_BINARY(fmax, fmaxf, fmax)

#undef WWR_MATH_BINARY

// The three-argument fused form, same dispatch shape. fma computes x*y + z as a
// single rounding -- the vendor intrinsic, not a bare `x * y + z`, is what keeps
// the fusion (and the extra precision) the caller asked for; the compiler is
// free to break the latter back into two roundings under -ffp-contract=off.
#define WWR_MATH_TERNARY(fn, ffn, dfn)                                                               \
  template <typename T> __device__ __forceinline__ T fn(const T x, const T y, const T z) {           \
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                             \
                  "wwr::" #fn " supports only float and double");                                    \
    if constexpr (std::is_same_v<T, float>) {                                                        \
      return ::ffn(x, y, z);                                                                         \
    } else {                                                                                         \
      return ::dfn(x, y, z);                                                                         \
    }                                                                                                \
  }

WWR_MATH_TERNARY(fma, fmaf, fma)

#undef WWR_MATH_TERNARY

// Classification predicates -- T in, bool out, so they need their own shape
// rather than the T->T macros above. Unlike the math functions, these have no
// f-suffixed spelling: the vendor supplies a single overloaded ::isnan (etc.)
// that resolves for float and double alike, so there is one entry point, not a
// float/double pair. The static_assert still fences the wrapper to the two types
// this file supports.
#define WWR_MATH_PREDICATE(fn, vfn)                                                                 \
  template <typename T> __device__ __forceinline__ bool fn(const T x) {                             \
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                            \
                  "wwr::" #fn " supports only float and double");                                   \
    return ::vfn(x);                                                                                \
  }

WWR_MATH_PREDICATE(isnan, isnan)
WWR_MATH_PREDICATE(isinf, isinf)
WWR_MATH_PREDICATE(isfinite, isfinite)

#undef WWR_MATH_PREDICATE

} // namespace wwr
