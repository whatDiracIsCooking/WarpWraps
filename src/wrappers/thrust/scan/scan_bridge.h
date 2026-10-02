/**
 * @file scan_bridge.h
 * @brief Declarations shared between wwr.wrappers.thrust.scan's interface unit
 *        and its device-compiled translation unit (the A-scan prefix-sum family)
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by scan.cu
 * directly. The declarations must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what scan.cu is (docs/architecture.md §14). The
 * host wrapper calls these; the device TU defines them and carries the explicit
 * template instantiations (§8).
 *
 * The `_bridge` suffix marks a header that crosses the host/device boundary, so
 * it must compile in BOTH modes -- hence a plain `.h`, not a `.cuh`. Its one
 * type, wwrStream_t, comes from runtime.h's always-on section (a GMF cannot
 * `import`), which is the SAME type wwr.runtime_api exports, so the wrapper
 * passes its stream straight through and the device side needs no cast
 * (§14, same idiom as init_state_bridge.h).
 *
 * Each entry point is a function TEMPLATE on the element type T; the device TU
 * instantiates exactly the portable type set (see scan.cu). The scan is over
 * the element's own `operator+` (thrust::plus<T>); the *_by_op variant takes
 * thrust::maximum<T> instead, the acceptance's custom-associative-operator path.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace wwr::thrust::device {

/// @brief out[0,n) = running sum (inclusive) of in[0,n), on `stream`
template<typename T>
void inclusive_scan(wwrStream_t stream, const T *in, std::size_t n, T *out);

/// @brief out[0,n) = running maximum (inclusive) of in[0,n), on `stream`
///
/// The custom-associative-operator path: thrust::maximum<T> in place of the
/// default sum. Real types only (a running max needs an ordering).
template<typename T>
void inclusive_scan_max(wwrStream_t stream, const T *in, std::size_t n, T *out);

/// @brief out[0,n) = exclusive prefix sum of in[0,n) seeded with `init`, on `stream`
///
/// out[0] = init, out[i] = init + sum(in[0,i)). `init` is passed by value.
template<typename T>
void exclusive_scan(wwrStream_t stream, const T *in, std::size_t n, T *out, T init);

/// @brief out[0,n) = inclusive prefix sum of negate(in[0,n)), on `stream`
///
/// The transform form: a unary op (thrust::negate<T>) applied before the sum.
template<typename T>
void transform_inclusive_scan_negate(wwrStream_t stream, const T *in, std::size_t n, T *out);

/// @brief out[0,n) = exclusive prefix sum of negate(in[0,n)) seeded with `init`, on `stream`
template<typename T>
void transform_exclusive_scan_negate(wwrStream_t stream, const T *in, std::size_t n, T *out,
                                     T init);

} // namespace wwr::thrust::device
