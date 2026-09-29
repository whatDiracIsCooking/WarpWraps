/**
 * @file complex.cppm
 * @brief wwr::complex<T> -- an ergonomic, operator-carrying value type over the
 *        gpu* complex types, for host TUs
 *
 * The gpu* complex layer (wwr.complex / complex.cuh) is deliberately
 * operator-less: cuComplex is a bare float2 aggregate where hipComplex is a
 * class, so `a * b` and brace-initialisation are not portable and arithmetic
 * lives in the wwrC* functions (docs/architecture.md, section 3). This wrapper
 * buys back the ergonomics on a type we own: complex<float>{1, 2} is plain
 * aggregate init, `+ - * /` are hidden friends computed on the components, and
 * the value implicitly converts to the vendor type (wwrFloatComplex /
 * wwrDoubleComplex) at a call site that expects one -- so it "evaluates to
 * wwrComplex".
 *
 * T is constrained to wwr::real_fp and mapped to the vendor type through
 * wwr::RealToComplexType, both from wwr.wrappers.common -- the same real/complex
 * vocabulary the blas/solver/sparse/fft wrappers are written against. The device
 * mirror (complex.cuh) cannot import that module, so it re-defines real_fp
 * locally; that is the one asymmetry between the two copies.
 *
 * The struct is an aggregate (public re/im, no declared constructors): the
 * conversion operator and the hidden-friend operators are non-constructor
 * members/free functions, so brace-init and the constant-expression narrowing
 * exception (complex<float>{1.0, 3.0} compiles; a runtime double would not) are
 * preserved. Layout is {T, T}, matching the vendor type, so to_complex round-
 * trips a vendor value back without a reinterpret_cast.
 *
 * The device-compile counterpart is device_complex.cuh: the SAME struct, the
 * SAME bodies, __device__ __forceinline__ where this says inline (the component
 * arithmetic is constexpr on both sides). Both reach make_gpu*Complex /
 * wwrCreal* by the same spelling -- this module through import wwr.complex, the
 * header through include complex.cuh -- so a kernel and a host TU agree on
 * layout.
 *
 * Usage:
 *   import wwr.wrappers.complex;
 *   using namespace wwr;
 *
 *   const complex<float> z = complex<float>{1.0f, 2.0f} * complex<float>{3.0f, -1.0f};
 *   const wwrFloatComplex v = z;            // implicit, for a BLAS/FFT call site
 *   const complex<float> back = to_complex(v);
 *
 * Note: with both `using namespace wwr;` and `using namespace std;` in scope,
 * unqualified `complex` is ambiguous with std::complex -- qualify wwr::complex.
 */

export module wwr.wrappers.complex;

import std;
import wwr.complex;
import wwr.wrappers.common;

export namespace wwr {

// ========================================================================
// complex<T>
//
// An aggregate value type: {re, im}, both T. real_fp constrains T to float or
// double -- the two precisions the gpu* complex layer names -- and
// RealToComplexType maps it to the vendor type (wwrFloatComplex /
// wwrDoubleComplex). Both come from wwr.wrappers.common.
// ========================================================================

template<real_fp T>
struct complex {
  /// The gpu* complex type this value converts to: float -> wwrFloatComplex,
  /// double -> wwrDoubleComplex.
  using vendor_type = RealToComplexType<T>;

  T re;
  T im;

  /// @brief Convert to the gpu* vendor complex type, through the portable
  ///        make_gpu*Complex (not a reinterpret_cast: same layout, distinct
  ///        type). Implicit, so a complex<T> drops straight into a call site
  ///        that expects wwrFloatComplex / wwrDoubleComplex.
  operator vendor_type() const {
    if constexpr (std::is_same_v<T, float>) {
      return make_gpuFloatComplex(re, im);
    } else {
      return make_gpuDoubleComplex(re, im);
    }
  }

  // Compound assignment as members (they mutate *this; a member does not
  // disqualify the aggregate). Scalar overloads treat T as the real axis.
  constexpr complex &operator+=(const complex b) { re += b.re; im += b.im; return *this; }
  constexpr complex &operator-=(const complex b) { re -= b.re; im -= b.im; return *this; }
  constexpr complex &operator*=(const complex b) { return *this = *this * b; }
  constexpr complex &operator/=(const complex b) { return *this = *this / b; }
  constexpr complex &operator+=(const T s) { re += s; return *this; }
  constexpr complex &operator-=(const T s) { re -= s; return *this; }
  constexpr complex &operator*=(const T s) { re *= s; im *= s; return *this; }
  constexpr complex &operator/=(const T s) { re /= s; im /= s; return *this; }

  // Arithmetic and comparison as hidden friends computed on the components:
  // constexpr and vendor-free (a declared friend does not disqualify the
  // aggregate). The vendor wwrC* functions remain the spelling for a raw
  // wwrFloatComplex. Scalar overloads treat T as the real axis.

  /// @brief Negation
  friend constexpr complex operator-(const complex a) { return {-a.re, -a.im}; }

  /// @brief Componentwise sum
  friend constexpr complex operator+(const complex a, const complex b) {
    return {a.re + b.re, a.im + b.im};
  }

  /// @brief Componentwise difference
  friend constexpr complex operator-(const complex a, const complex b) {
    return {a.re - b.re, a.im - b.im};
  }

  /// @brief Complex product
  friend constexpr complex operator*(const complex a, const complex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
  }

  /// @brief Complex quotient
  friend constexpr complex operator/(const complex a, const complex b) {
    const T denom = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom};
  }

  /// @brief Equality -- exact, componentwise; != is synthesised from it
  friend constexpr bool operator==(const complex a, const complex b) {
    return a.re == b.re && a.im == b.im;
  }

  /// @brief Scaling by a real, either order
  friend constexpr complex operator*(const complex a, const T s) { return {a.re * s, a.im * s}; }
  friend constexpr complex operator*(const T s, const complex a) { return {a.re * s, a.im * s}; }

  /// @brief Division by a real
  friend constexpr complex operator/(const complex a, const T s) { return {a.re / s, a.im / s}; }

  /// @brief Offset along the real axis, either order
  friend constexpr complex operator+(const complex a, const T s) { return {a.re + s, a.im}; }
  friend constexpr complex operator+(const T s, const complex a) { return {a.re + s, a.im}; }
  friend constexpr complex operator-(const complex a, const T s) { return {a.re - s, a.im}; }
  friend constexpr complex operator-(const T s, const complex a) { return {s - a.re, -a.im}; }
};

// ========================================================================
// Reverse conversion
//
// Read a gpu* vendor value back into a complex<T>, through the portable
// wwrCreal* / wwrCimag* accessors (the vendor types expose no portable .x/.y).
// Two overloads: wwrFloatComplex (== wwrComplex) and wwrDoubleComplex.
// ========================================================================

/// @brief Build a complex<float> from a single-precision vendor value
inline complex<float> to_complex(const wwrFloatComplex z) {
  return {wwrCrealf(z), wwrCimagf(z)};
}

/// @brief Build a complex<double> from a double-precision vendor value
inline complex<double> to_complex(const wwrDoubleComplex z) {
  return {wwrCreal(z), wwrCimag(z)};
}

} // namespace wwr
