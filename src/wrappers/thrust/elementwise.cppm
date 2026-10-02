/**
 * @file elementwise.cppm
 * @brief Primary interface for wwr.wrappers.thrust -- the elementwise family
 *
 * Typed, backend-neutral host wrappers over the Thrust/rocThrust elementwise
 * algorithms, dispatched through the F2 stream-bound execution policy so each
 * runs on the caller's stream (issue #241). The actual thrust:: calls live in
 * elementwise.cu, device-compiled for the selected backend; these host wrappers
 * forward to the device:: free functions it defines (the host/device-link
 * boundary, docs/architecture.md §8/§14, as in init_state / random_normal).
 *
 * This is the module-altitude surface: the ops whose shape needs NO
 * caller-supplied functor -- `fill`, `sequence`, `replace`, and `transform`
 * over a fixed portable op set (negate / square / abs; plus / minus / multiply).
 * Each is instantiated once (extern template below, definition in the .cu), so
 * an importer links rather than re-instantiating a Thrust call tree per site.
 *
 * `for_each`, `generate`, `tabulate` and a free-functor `transform` -- the forms
 * whose whole point is an arbitrary caller op, which cannot cross a host/device
 * link boundary -- live instead in the header template algorithms.cuh, which a
 * caller's own device TU #includes and instantiates. And the lower-level,
 * hand-written index-per-thread kernel is src/extension/parallel_for; this
 * family does NOT replace it. See README.md, "transform vs parallel_for".
 *
 * Element-type coverage is per-algorithm and justified where it narrows:
 *   fill / replace          -- all numeric, complex included
 *   sequence                -- reals + integers (no linear step for complex)
 *   transform (negate/sq.)  -- all numeric, complex included
 *   transform (abs)         -- reals + signed integers only
 *   transform (binary)      -- all numeric, complex included
 *
 * Usage:
 *   import wwr.wrappers.thrust;
 *   using namespace wwr::extension::thrust;
 *   fill(stream, d_out, n, 1.0f);
 *   transform_unary<UnaryOp::Negate>(stream, d_in, d_out, n);
 */

module;

#include "wrappers/thrust/elementwise_bridge.h"

export module wwr.wrappers.thrust;

import std;
import wwr.runtime_api;
import wwr.complex;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each wrapper carries its own
// `export` and the extern templates sit in the plain namespace -- the same
// shape wwr.extension.random_normal uses.
namespace wwr::extension::thrust {

// Re-export the op selectors (declared in elementwise_bridge.h, in this same
// namespace but in the GLOBAL MODULE FRAGMENT -- a GMF name is not part of the
// module's interface, so an importer sees it only through this export). A
// using-declaration naming a type is an exportable redeclaration of it.
export using wwr::extension::thrust::UnaryOp;
export using wwr::extension::thrust::BinaryOp;

/// @brief Set every element of `[out, out+count)` to `value` (thrust::fill).
/// @param stream Stream the work is enqueued on (async)
/// @param out    Device array of at least `count` elements
export template<typename T>
void fill(const wwrStream_t stream, T *out, const std::size_t count, const T value) {
  device::fill(stream, out, count, value);
}

/// @brief Write the arithmetic progression `init, init+step, ...` (thrust::sequence).
/// @note Reals and integers only; complex has no meaningful linear step.
export template<typename T>
void sequence(const wwrStream_t stream, T *out, const std::size_t count, const T init = T{0},
              const T step = T{1}) {
  device::sequence(stream, out, count, init, step);
}

/// @brief Replace every element equal to `old_value` with `new_value` (thrust::replace).
export template<typename T>
void replace(const wwrStream_t stream, T *out, const std::size_t count, const T old_value,
             const T new_value) {
  device::replace(stream, out, count, old_value, new_value);
}

/// @brief Apply a portable unary op elementwise: `out[i] = Op(in[i])` (thrust::transform).
/// @tparam Op One of UnaryOp; the op is selected at compile time, no functor crosses the boundary
export template<UnaryOp Op, typename T>
void transform_unary(const wwrStream_t stream, const T *in, T *out, const std::size_t count) {
  device::transform_unary<Op>(stream, in, out, count);
}

/// @brief Apply a portable binary op elementwise: `out[i] = Op(a[i], b[i])` (thrust::transform).
export template<BinaryOp Op, typename T>
void transform_binary(const wwrStream_t stream, const T *a, const T *b, T *out,
                       const std::size_t count) {
  device::transform_binary<Op>(stream, a, b, out, count);
}

// ------------------------------------------------------------------------
// Instantiated once in instantiations.cpp, not at every call site. The
// definitions these resolve to are in elementwise.cu, compiled as device code.
// Kept in step with elementwise.cu's and instantiations.cpp's lists.
// ------------------------------------------------------------------------

#define WWR_X_FILL(T) extern template void fill<T>(wwrStream_t, T *, std::size_t, T)
WWR_X_FILL(float);
WWR_X_FILL(double);
WWR_X_FILL(int);
WWR_X_FILL(unsigned int);
WWR_X_FILL(long long);
WWR_X_FILL(unsigned long long);
WWR_X_FILL(wwrFloatComplex);
WWR_X_FILL(wwrDoubleComplex);
#undef WWR_X_FILL

#define WWR_X_SEQ(T) extern template void sequence<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_X_SEQ(float);
WWR_X_SEQ(double);
WWR_X_SEQ(int);
WWR_X_SEQ(unsigned int);
WWR_X_SEQ(long long);
WWR_X_SEQ(unsigned long long);
#undef WWR_X_SEQ

#define WWR_X_REP(T) extern template void replace<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_X_REP(float);
WWR_X_REP(double);
WWR_X_REP(int);
WWR_X_REP(unsigned int);
WWR_X_REP(long long);
WWR_X_REP(unsigned long long);
WWR_X_REP(wwrFloatComplex);
WWR_X_REP(wwrDoubleComplex);
#undef WWR_X_REP

#define WWR_X_UN(Op, T) \
  extern template void transform_unary<Op, T>(wwrStream_t, const T *, T *, std::size_t)
WWR_X_UN(UnaryOp::Negate, float);
WWR_X_UN(UnaryOp::Negate, double);
WWR_X_UN(UnaryOp::Negate, int);
WWR_X_UN(UnaryOp::Negate, long long);
WWR_X_UN(UnaryOp::Negate, wwrFloatComplex);
WWR_X_UN(UnaryOp::Negate, wwrDoubleComplex);
WWR_X_UN(UnaryOp::Square, float);
WWR_X_UN(UnaryOp::Square, double);
WWR_X_UN(UnaryOp::Square, int);
WWR_X_UN(UnaryOp::Square, long long);
WWR_X_UN(UnaryOp::Square, wwrFloatComplex);
WWR_X_UN(UnaryOp::Square, wwrDoubleComplex);
WWR_X_UN(UnaryOp::Abs, float);
WWR_X_UN(UnaryOp::Abs, double);
WWR_X_UN(UnaryOp::Abs, int);
WWR_X_UN(UnaryOp::Abs, long long);
#undef WWR_X_UN

#define WWR_X_BIN(Op, T) \
  extern template void transform_binary<Op, T>(wwrStream_t, const T *, const T *, T *, std::size_t)
#define WWR_X_BIN_ALL(Op)  \
  WWR_X_BIN(Op, float);    \
  WWR_X_BIN(Op, double);   \
  WWR_X_BIN(Op, int);      \
  WWR_X_BIN(Op, long long); \
  WWR_X_BIN(Op, wwrFloatComplex); \
  WWR_X_BIN(Op, wwrDoubleComplex)
WWR_X_BIN_ALL(BinaryOp::Plus);
WWR_X_BIN_ALL(BinaryOp::Minus);
WWR_X_BIN_ALL(BinaryOp::Multiply);
#undef WWR_X_BIN_ALL
#undef WWR_X_BIN

} // namespace wwr::extension::thrust
