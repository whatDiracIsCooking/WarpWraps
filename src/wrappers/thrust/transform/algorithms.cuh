/// @file
/// @brief Backend-neutral header-template elementwise algorithms for a device TU
///
/// The functor-taking half of the A-transform family (issue #241): `for_each`,
/// `generate`, `tabulate`, and a free-functor `transform` (unary and binary).
/// Their whole point is an ARBITRARY caller-supplied op -- a device functor type
/// that cannot cross a host/device link boundary, so unlike the fixed-op surface
/// in wwr.wrappers.thrust.transform (interface.cppm) these cannot be
/// pre-instantiated in a .cu and exposed through a module. They are header
/// templates instead: a caller's own `.cu` (CUDA) or `-x hip` device-compiled TU
/// `#include`s this and instantiates them with its functor, exactly as
/// parallel_for.cuh is consumed.
///
/// Each forwards to `thrust::<algo>(wwr::par_on(stream), ...)`, so the work lands
/// on the caller's stream; the F2 README audit confirms every overload here is
/// byte-identical across CCCL Thrust 3.0.1 and rocThrust 2.8.5.
///
/// transform vs parallel_for (src/extension/parallel_for/parallel_for.cuh):
/// reach for these when the work is a clean elementwise map over contiguous
/// device buffer(s) and you want Thrust's range/iterator ergonomics; reach for
/// parallel_for for the lower-level, hand-written index-per-thread kernel --
/// manual indexing, multi-array gather/scatter, or non-elementwise work. This
/// family does NOT replace parallel_for; see README.md.
///
/// `#include`d directly into a device TU, never a module -- the Thrust headers
/// assume a device pass and a device TU cannot `import` (§8). Link wwr::thrust
/// (Thrust include dirs, the src/ root, wwr::device's runtime). The src/-root
/// spelling "wrappers/thrust/transform/algorithms.cuh" resolves off that same
/// root.
#pragma once

// execution_policy.cuh first: it pulls device_guard.h (the device-pass gate),
// runtime.h (wwrStream_t), and the _VSTD / hipStreamDefault repairs rocThrust
// needs -- all before any <thrust/*> header.
#include "wrappers/thrust/execution_policy.cuh"

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cstddef>

namespace wwr::extension::thrust {

namespace detail {

/// @brief Wrap a raw device pointer so Thrust dispatches on device memory.
template<typename T>
auto dptr(T *p) {
  return ::thrust::device_pointer_cast(p);
}

} // namespace detail

/// @brief Invoke `f(in[i])` for each element, on `stream` (thrust::for_each).
/// @tparam F A device-callable functor (its `__device__ operator()` takes a T)
template<typename T, typename F>
void for_each(const wwrStream_t stream, T *data, const std::size_t count, F f) {
  if (count < 1) {
    return;
  }
  const auto first = detail::dptr(data);
  ::thrust::for_each(wwr::par_on(stream), first, first + count, f);
}

/// @brief Write `out[i] = gen()` for each element, on `stream` (thrust::generate).
/// @tparam Gen A device-callable nullary generator returning a T
template<typename T, typename Gen>
void generate(const wwrStream_t stream, T *out, const std::size_t count, Gen gen) {
  if (count < 1) {
    return;
  }
  const auto first = detail::dptr(out);
  ::thrust::generate(wwr::par_on(stream), first, first + count, gen);
}

/// @brief Write `out[i] = op(i)` for each index, on `stream` (thrust::tabulate).
/// @tparam Op A device-callable unary functor mapping an index to a T
template<typename T, typename Op>
void tabulate(const wwrStream_t stream, T *out, const std::size_t count, Op op) {
  if (count < 1) {
    return;
  }
  const auto first = detail::dptr(out);
  ::thrust::tabulate(wwr::par_on(stream), first, first + count, op);
}

/// @brief Apply a caller unary op elementwise: `out[i] = op(in[i])` (thrust::transform).
/// @tparam Op A device-callable unary functor
template<typename T, typename Op>
void transform(const wwrStream_t stream, const T *in, T *out, const std::size_t count, Op op) {
  if (count < 1) {
    return;
  }
  const auto first = detail::dptr(in);
  ::thrust::transform(wwr::par_on(stream), first, first + count, detail::dptr(out), op);
}

/// @brief Apply a caller binary op elementwise: `out[i] = op(a[i], b[i])` (thrust::transform).
/// @tparam Op A device-callable binary functor
template<typename T, typename Op>
void transform(const wwrStream_t stream, const T *a, const T *b, T *out, const std::size_t count,
               Op op) {
  if (count < 1) {
    return;
  }
  const auto first = detail::dptr(a);
  ::thrust::transform(wwr::par_on(stream), first, first + count, detail::dptr(b), detail::dptr(out),
                      op);
}

} // namespace wwr::extension::thrust
