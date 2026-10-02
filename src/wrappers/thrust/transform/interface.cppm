/**
 * @file interface.cppm
 * @brief Primary interface for wwr.wrappers.thrust.transform -- the elementwise
 *        (A-transform) family
 *
 * Typed, backend-neutral host wrappers over the Thrust/rocThrust elementwise
 * algorithms, dispatched through the F2 stream-bound execution policy so each
 * runs on the caller's stream (issue #241). The actual thrust:: calls live in
 * transform.cu, device-compiled for the selected backend; these host wrappers
 * forward to the device:: free functions it defines (the host/device-link
 * boundary, docs/architecture.md §8/§14, as in init_state / random_normal).
 * Each forwarder only emits a device:: call, so -- like wwr.wrappers.thrust.scan
 * -- it instantiates at the call site without a separate implementation unit;
 * the heavy Thrust call tree is pre-instantiated once in transform.cu.
 *
 * This is the module-altitude surface: the ops whose shape needs NO
 * caller-supplied functor -- `fill`, `sequence`, `replace`, and `transform`
 * over a fixed portable op set (negate / square / abs; plus / minus / multiply).
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
 *   import wwr.wrappers.thrust.transform;
 *   using namespace wwr::extension::thrust;
 *   fill(stream, d_out, n, 1.0f);
 *   transform_unary<UnaryOp::Negate>(stream, d_in, d_out, n);
 */

module;

#include "wrappers/thrust/transform/transform_bridge.h"

export module wwr.wrappers.thrust.transform;

import std;
import wwr.runtime_api;
import wwr.complex;

export namespace wwr::extension::thrust {

// Re-export the op selectors (declared in transform_bridge.h, in this same
// namespace but in the GLOBAL MODULE FRAGMENT -- a GMF name is not part of the
// module's interface, so an importer sees it only through this export). A
// using-declaration naming a type is an exportable redeclaration of it.
using wwr::extension::thrust::BinaryOp;
using wwr::extension::thrust::UnaryOp;

/// @brief Set every element of `[out, out+count)` to `value` (thrust::fill).
/// @param stream Stream the work is enqueued on (async)
/// @param out    Device array of at least `count` elements
template<typename T>
void fill(const wwrStream_t stream, T *out, const std::size_t count, const T value) {
  device::fill(stream, out, count, value);
}

/// @brief Write the arithmetic progression `init, init+step, ...` (thrust::sequence).
/// @note Reals and integers only; complex has no meaningful linear step.
template<typename T>
void sequence(const wwrStream_t stream, T *out, const std::size_t count, const T init = T{0},
              const T step = T{1}) {
  device::sequence(stream, out, count, init, step);
}

/// @brief Replace every element equal to `old_value` with `new_value` (thrust::replace).
template<typename T>
void replace(const wwrStream_t stream, T *out, const std::size_t count, const T old_value,
             const T new_value) {
  device::replace(stream, out, count, old_value, new_value);
}

/// @brief Apply a portable unary op elementwise: `out[i] = Op(in[i])` (thrust::transform).
/// @tparam Op One of UnaryOp; the op is selected at compile time, no functor crosses the boundary
template<UnaryOp Op, typename T>
void transform_unary(const wwrStream_t stream, const T *in, T *out, const std::size_t count) {
  device::transform_unary<Op>(stream, in, out, count);
}

/// @brief Apply a portable binary op elementwise: `out[i] = Op(a[i], b[i])` (thrust::transform).
template<BinaryOp Op, typename T>
void transform_binary(const wwrStream_t stream, const T *a, const T *b, T *out,
                       const std::size_t count) {
  device::transform_binary<Op>(stream, a, b, out, count);
}

} // namespace wwr::extension::thrust
