/**
 * @file reduce_bridge.h
 * @brief Host/device boundary for wwr.wrappers.thrust.reduce
 *
 * The reduction family's typed entry points, declared once here so the module
 * interface (interface.cppm, in its GLOBAL MODULE FRAGMENT) and the
 * device-compiled translation unit (reduce.cu) name the same functions. Each is
 * a function TEMPLATE over the element type `T`; reduce.cu both defines the
 * primary template and emits an explicit instantiation per supported `T`, so
 * the interface's matching `extern template` declarations resolve against a
 * definition compiled in a plain (non-module) TU -- exactly the linkage
 * arrangement docs/architecture.md §8 and §14 describe, and the same
 * declaration-in-the-GMF rule init_state_bridge.h follows.
 *
 * A GMF cannot `import`, so wwrStream_t arrives by #include (runtime.h), the
 * SAME type wwr.runtime_api exports -- the wrapper passes it straight through
 * with no cast. The `_bridge` suffix marks a header that must compile in BOTH
 * host and device mode; a plain `.h` is the only spelling that can (a `.cuh`
 * is device-pass-only). See src/README.md.
 *
 * Why fixed operators rather than caller-supplied functors: a device functor
 * cannot cross from a host TU into a device launch, so each entry here binds
 * the one portable, default-constructible operator its algorithm needs
 * (`thrust::plus` for the sums, `thrust::square`-equivalent for the transform,
 * equality/non-zero for the counts, `<` for the extrema) inside reduce.cu. The
 * result is returned BY VALUE: the reduction's scalar answer is copied back to
 * the host, which is the typed, allocation-free shape a caller wants.
 */

#pragma once

#include "runtime.h"

#include <cstddef>
#include <cstdint>

namespace wwr::thrust::device {

/// @brief sum over [first, first + n), seeded with `init` (thrust::reduce)
template<typename T>
T reduce_sum(wwrStream_t stream, const T *first, std::size_t n, T init);

/// @brief sum of squares over [first, first + n), seeded with `init`
///        (thrust::transform_reduce, unary = x*x, binary = +)
template<typename T>
T transform_reduce_square_sum(wwrStream_t stream, const T *first, std::size_t n, T init);

/// @brief number of elements in [first, first + n) equal to `value`
///        (thrust::count)
template<typename T>
std::int64_t count(wwrStream_t stream, const T *first, std::size_t n, T value);

/// @brief number of non-zero elements in [first, first + n)
///        (thrust::count_if with a `!= T{}` predicate)
template<typename T>
std::int64_t count_if_nonzero(wwrStream_t stream, const T *first, std::size_t n);

/// @brief dot product of two length-n device ranges, seeded with `init`
///        (thrust::inner_product)
template<typename T>
T inner_product(wwrStream_t stream, const T *first1, const T *first2, std::size_t n, T init);

/// @brief value of the smallest element in [first, first + n)
///        (thrust::min_element, dereferenced). `n` must be >= 1.
template<typename T>
T min_element_value(wwrStream_t stream, const T *first, std::size_t n);

/// @brief value of the largest element in [first, first + n)
///        (thrust::max_element, dereferenced). `n` must be >= 1.
template<typename T>
T max_element_value(wwrStream_t stream, const T *first, std::size_t n);

} // namespace wwr::thrust::device
