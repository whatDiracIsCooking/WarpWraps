/**
 * @file interface.cppm
 * @brief Primary interface for wwr.wrappers.thrust.scan (A-scan: prefix sums)
 *
 * Typed, backend-neutral wrappers for the Thrust prefix-sum family --
 * inclusive_scan, exclusive_scan, transform_inclusive_scan and
 * transform_exclusive_scan -- over device memory, each running on a caller-
 * supplied stream. Like the src/wrappers/blas layer these are thin typed
 * forwarders; unlike it, the work cannot be done from a module unit, because
 * Thrust is device-pass-only (docs/architecture.md §8). So the actual
 * thrust::*_scan(wwr::par_on(stream), ...) calls live in scan.cu, a device TU,
 * and this module forwards to the wwr::thrust::device:: entry points it declares
 * through scan_bridge.h -- the host/device link boundary (§14), as in
 * wwr.extension.init_state.
 *
 * Type coverage is arithmetic (float, double, int, std::int64_t); the device
 * library instantiates exactly that set. See scan.cu for why complex is out of
 * scope here.
 *
 * Usage:
 *   import wwr.wrappers.thrust.scan;
 *   using namespace wwr::thrust;
 *
 *   // d_in, d_out are device pointers; stream is a wwrStream_t
 *   inclusive_scan(stream, d_in, n, d_out);        // running sum
 *   exclusive_scan(stream, d_in, n, d_out, 0.0f);  // seeded exclusive sum
 *   inclusive_scan(stream, d_in, n, d_out, Op::max);  // custom associative op
 */

module;

#include "wrappers/thrust/scan/scan_bridge.h"

export module wwr.wrappers.thrust.scan;

import std;
import wwr.runtime_api;

export namespace wwr::thrust {

/// @brief Element types this family supports: the arithmetic scalars.
///
/// Matches scan.cu's explicit instantiations exactly. A call with any other T
/// is a compile error (the concept), not a link error, which is the point of
/// stating the set here rather than leaving it to the linker.
template<typename T>
concept scan_scalar = std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, int> || std::is_same_v<T, std::int64_t>;

/// @brief Selects the associative operator an inclusive_scan folds with.
///
/// `plus` is the prefix-sum default; `max` is the acceptance's custom-operator
/// path (a running maximum). Named rather than a functor parameter because the
/// device TU can only instantiate concrete operators (§8).
enum class Op { plus, max };

/**
 * @brief Inclusive prefix scan: out[i] = op(in[0], ..., in[i]), on `stream`
 *
 * out[0] = in[0]; out may alias in (Thrust's inclusive_scan supports in-place).
 *
 * @param stream Stream the scan's work is enqueued on (asynchronous)
 * @param in Device input array of at least `n` elements
 * @param n Number of elements
 * @param out Device output array of at least `n` elements; may equal `in`
 * @param op Associative operator to fold with (default sum)
 */
template<scan_scalar T>
void inclusive_scan(const wwrStream_t stream, const T *in, const std::size_t n, T *out,
                    const Op op = Op::plus) {
  // device:: is load-bearing -- without it this names itself.
  if (op == Op::max) {
    device::inclusive_scan_max(stream, in, n, out);
  } else {
    device::inclusive_scan(stream, in, n, out);
  }
}

/**
 * @brief Exclusive prefix sum seeded with `init`, on `stream`
 *
 * out[0] = init, out[i] = init + in[0] + ... + in[i-1]. out may alias in.
 *
 * @param stream Stream the scan's work is enqueued on (asynchronous)
 * @param in Device input array of at least `n` elements
 * @param n Number of elements
 * @param out Device output array of at least `n` elements; may equal `in`
 * @param init Value the exclusive sum starts from (out[0])
 */
template<scan_scalar T>
void exclusive_scan(const wwrStream_t stream, const T *in, const std::size_t n, T *out,
                    const T init) {
  device::exclusive_scan(stream, in, n, out, init);
}

/**
 * @brief Inclusive scan of the negated input: out[i] = -in[0] - ... - in[i]
 *
 * The transform form -- a unary op (negate) applied to each element before the
 * inclusive sum. Demonstrates the transform_inclusive_scan overload over the
 * arithmetic set.
 *
 * @param stream Stream the scan's work is enqueued on (asynchronous)
 * @param in Device input array of at least `n` elements
 * @param n Number of elements
 * @param out Device output array of at least `n` elements; may equal `in`
 */
template<scan_scalar T>
void transform_inclusive_scan_negate(const wwrStream_t stream, const T *in, const std::size_t n,
                                     T *out) {
  device::transform_inclusive_scan_negate(stream, in, n, out);
}

/**
 * @brief Exclusive scan of the negated input seeded with `init`, on `stream`
 *
 * out[0] = init, out[i] = init - in[0] - ... - in[i-1].
 *
 * @param stream Stream the scan's work is enqueued on (asynchronous)
 * @param in Device input array of at least `n` elements
 * @param n Number of elements
 * @param out Device output array of at least `n` elements; may equal `in`
 * @param init Value the exclusive sum starts from (out[0])
 */
template<scan_scalar T>
void transform_exclusive_scan_negate(const wwrStream_t stream, const T *in, const std::size_t n,
                                     T *out, const T init) {
  device::transform_exclusive_scan_negate(stream, in, n, out, init);
}

} // namespace wwr::thrust
