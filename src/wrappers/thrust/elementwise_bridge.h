/**
 * @file elementwise_bridge.h
 * @brief Declarations shared between the elementwise module's interface units
 *        and its device-compiled translation unit
 *
 * Included by elementwise.cppm in its GLOBAL MODULE FRAGMENT, and by
 * elementwise.cu directly. The `_bridge` suffix marks a header that carries a
 * declaration across the host/device boundary, so it must compile in BOTH modes
 * (plain host pass for the module, device pass for the .cu) -- see src/README.md
 * and docs/architecture.md §14 for why the device declarations live here, in the
 * GMF, rather than in the module purview: a purview name gets module linkage and
 * can never bind to the definition compiled in elementwise.cu, a plain TU.
 *
 * The named-op selectors (UnaryOp / BinaryOp) are plain scoped enums usable from
 * host and device alike, so the host wrapper takes one as a compile-time
 * template argument and the .cu maps it to the concrete device functor. The
 * device-side free functions the exported wrappers forward to live in the nested
 * `device` namespace below, exactly as wwr.extension.random_normal splits its
 * device:: half out.
 *
 * wwrStream_t comes from runtime.h (an #include-only header, since a GMF cannot
 * import) -- the SAME type wwr.runtime_api exports, so the wrapper passes its
 * stream straight through and the device side needs no cast (§8).
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace wwr::extension::thrust {

/// @brief Portable unary elementwise operations for transform.
///
/// Each maps to a device functor in elementwise.cu built from the vendor-neutral
/// `wwr*` device primitives (so complex uses `wwrC*`, not operators cuComplex
/// lacks). The set is deliberately small and value-free: an arbitrary
/// caller-supplied functor cannot cross this host/device boundary (the device TU
/// must see the functor type at compile time), so callers needing a custom op
/// use the `algorithms.cuh` header templates or `parallel_for.cuh` instead.
enum class UnaryOp {
  Negate, ///< y = -x (all numeric, including complex)
  Abs,    ///< y = |x| (reals and signed integers only; complex magnitude is a different type)
  Square, ///< y = x * x (all numeric, including complex)
};

/// @brief Portable binary elementwise operations for transform.
enum class BinaryOp {
  Plus,     ///< z = x + y
  Minus,    ///< z = x - y
  Multiply, ///< z = x * y
};

namespace device {

/// @brief Fill `[out, out+count)` with `value` via thrust::fill on `stream`.
template<typename T>
void fill(wwrStream_t stream, T *out, std::size_t count, T value);

/// @brief Write `init, init+step, init+2*step, ...` via thrust::sequence.
template<typename T>
void sequence(wwrStream_t stream, T *out, std::size_t count, T init, T step);

/// @brief Replace every element equal to `old_value` with `new_value`.
template<typename T>
void replace(wwrStream_t stream, T *out, std::size_t count, T old_value, T new_value);

/// @brief Apply the selected unary op elementwise: out[i] = op(in[i]).
template<UnaryOp Op, typename T>
void transform_unary(wwrStream_t stream, const T *in, T *out, std::size_t count);

/// @brief Apply the selected binary op elementwise: out[i] = op(a[i], b[i]).
template<BinaryOp Op, typename T>
void transform_binary(wwrStream_t stream, const T *a, const T *b, T *out, std::size_t count);

} // namespace device

} // namespace wwr::extension::thrust
