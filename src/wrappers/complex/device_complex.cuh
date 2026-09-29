/**
 * @file device_complex.cuh
 * @brief wwr::complex<T> for device-compiled TUs -- the mirror of
 *        wwr.wrappers.complex
 *
 * Named device_complex.cuh, not complex.cuh, on purpose: the gpu* layer it
 * includes is src/complex.cuh, and a same-basename header here would shadow it
 * under quoted-include's search-own-directory-first rule (the #include below
 * would resolve to this file). The distinct stem lets "complex.cuh" reach the
 * gpu* header.
 *
 * The device-compile counterpart to wrappers/complex/complex.cppm: the SAME
 * aggregate {re, im}, the SAME bodies, spelled __device__ __forceinline__ where
 * the module says inline. The component arithmetic is constexpr on both sides
 * (it folds in a device constant expression too); only the conversion operator
 * and to_complex are non-constexpr, because they route through the vendor
 * make_gpu*Complex / wwrCreal* functions, which are not. The two copies define
 * the same-layout type under the same name, so a complex<T> built on the host
 * and one named in a kernel agree -- the invariant a reinterpret_cast-free
 * conversion depends on. Link wwr.device.
 *
 * The one asymmetry: the module constrains T with wwr::real_fp imported from
 * wwr.wrappers.common; a header cannot import that module, so this file
 * re-defines the concept locally (kept identical -- see the note at its
 * definition below).
 *
 * The wrapper carries operators because it is a type we own; the gpu* complex
 * layer beneath (complex.cuh) stays operator-less on purpose, since cuComplex
 * is a float2 aggregate and hipComplex a class (docs/architecture.md,
 * section 3). This header reaches make_gpu*Complex and the wwrCreal* accessors
 * through that lower header by the same spelling the module reaches them by
 * import.
 *
 * __device__ __forceinline__, like complex.cuh / fp16.cuh: these serve a
 * parallel_for functor's __device__ operator(). Host code reaches the same
 * surface through the module, not this header.
 */

#pragma once

// The gpu* complex layer: device_guard.h (WWR_SELECTED_*, and #errors outside a
// device pass), the vendor types, make_gpu*Complex and the wwrCreal*/wwrCimag*
// accessors this wrapper is built on.
#include "complex.cuh"

#include <type_traits>

namespace wwr {

// The device-side redefinition of wwr.wrappers.common's real_fp
// (wrappers/common/fp_types.cppm): a header cannot import that module, so the
// concept the module constrains complex<T> with is spelled out again here. Kept
// identical on purpose -- if one moves, both must.
template<typename T>
concept real_fp = std::is_same_v<T, float> || std::is_same_v<T, double>;

// ========================================================================
// complex<T> -- the same aggregate wwr.wrappers.complex exports
// ========================================================================

template<real_fp T>
struct complex {
  /// The gpu* complex type this value converts to: float -> wwrFloatComplex,
  /// double -> wwrDoubleComplex.
  using vendor_type =
      std::conditional_t<std::is_same_v<T, float>, wwrFloatComplex, wwrDoubleComplex>;

  T re;
  T im;

  /// @brief Convert to the gpu* vendor complex type, through the portable
  ///        make_gpu*Complex (not a reinterpret_cast: same layout, distinct
  ///        type). Implicit, so a complex<T> drops straight into a call that
  ///        expects wwrFloatComplex / wwrDoubleComplex.
  __device__ __forceinline__ operator vendor_type() const {
    if constexpr (std::is_same_v<T, float>) {
      return make_gpuFloatComplex(re, im);
    } else {
      return make_gpuDoubleComplex(re, im);
    }
  }

  // Compound assignment as members (they mutate *this; a member does not
  // disqualify the aggregate). Scalar overloads treat T as the real axis.
  __device__ __forceinline__ constexpr complex &operator+=(const complex b) { re += b.re; im += b.im; return *this; }
  __device__ __forceinline__ constexpr complex &operator-=(const complex b) { re -= b.re; im -= b.im; return *this; }
  __device__ __forceinline__ constexpr complex &operator*=(const complex b) { return *this = *this * b; }
  __device__ __forceinline__ constexpr complex &operator/=(const complex b) { return *this = *this / b; }
  __device__ __forceinline__ constexpr complex &operator+=(const T s) { re += s; return *this; }
  __device__ __forceinline__ constexpr complex &operator-=(const T s) { re -= s; return *this; }
  __device__ __forceinline__ constexpr complex &operator*=(const T s) { re *= s; im *= s; return *this; }
  __device__ __forceinline__ constexpr complex &operator/=(const T s) { re /= s; im /= s; return *this; }

  /// @brief Negation
  friend __device__ __forceinline__ constexpr complex operator-(const complex a) { return {-a.re, -a.im}; }

  /// @brief Componentwise sum
  friend __device__ __forceinline__ constexpr complex operator+(const complex a, const complex b) {
    return {a.re + b.re, a.im + b.im};
  }

  /// @brief Componentwise difference
  friend __device__ __forceinline__ constexpr complex operator-(const complex a, const complex b) {
    return {a.re - b.re, a.im - b.im};
  }

  /// @brief Complex product
  friend __device__ __forceinline__ constexpr complex operator*(const complex a, const complex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
  }

  /// @brief Complex quotient
  friend __device__ __forceinline__ constexpr complex operator/(const complex a, const complex b) {
    const T denom = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom};
  }

  /// @brief Equality -- exact, componentwise; != is synthesised from it
  friend __device__ __forceinline__ constexpr bool operator==(const complex a, const complex b) {
    return a.re == b.re && a.im == b.im;
  }

  /// @brief Scaling by a real, either order
  friend __device__ __forceinline__ constexpr complex operator*(const complex a, const T s) { return {a.re * s, a.im * s}; }
  friend __device__ __forceinline__ constexpr complex operator*(const T s, const complex a) { return {a.re * s, a.im * s}; }

  /// @brief Division by a real
  friend __device__ __forceinline__ constexpr complex operator/(const complex a, const T s) { return {a.re / s, a.im / s}; }

  /// @brief Offset along the real axis, either order
  friend __device__ __forceinline__ constexpr complex operator+(const complex a, const T s) { return {a.re + s, a.im}; }
  friend __device__ __forceinline__ constexpr complex operator+(const T s, const complex a) { return {a.re + s, a.im}; }
  friend __device__ __forceinline__ constexpr complex operator-(const complex a, const T s) { return {a.re - s, a.im}; }
  friend __device__ __forceinline__ constexpr complex operator-(const T s, const complex a) { return {s - a.re, -a.im}; }
};

// ========================================================================
// Reverse conversion -- read a gpu* vendor value back into a complex<T>,
// through the portable wwrCreal* / wwrCimag* accessors.
// ========================================================================

/// @brief Build a complex<float> from a single-precision vendor value
__device__ __forceinline__ complex<float> to_complex(const wwrFloatComplex z) {
  return {wwrCrealf(z), wwrCimagf(z)};
}

/// @brief Build a complex<double> from a double-precision vendor value
__device__ __forceinline__ complex<double> to_complex(const wwrDoubleComplex z) {
  return {wwrCreal(z), wwrCimag(z)};
}

} // namespace wwr
