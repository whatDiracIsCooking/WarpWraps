// elementwise.cu
//
// The device half of wwr.wrappers.thrust's elementwise family. It is the real
// consumer of the F2 shim: a device TU that #includes execution_policy.cuh and
// calls thrust::<algo>(wwr::par_on(stream), ...), letting the backend policy
// (thrust::cuda::par on CUDA, thrust::hip::par on HIP) be selected from the
// compiler's own device-pass macro. Shared unchanged between both backends --
// .cu means "device pass, whichever backend" (docs/architecture.md §17), and the
// README audit confirms every in-scope overload is byte-identical across CCCL
// Thrust 3.0.1 and rocThrust 2.8.5, so no per-backend #if is needed here.
//
// execution_policy.cuh first: it pulls runtime.h (wwrStream_t for the bridge
// declarations) and the _VSTD / hipStreamDefault repairs the rocThrust headers
// need, both before any <thrust/*> include.
#include "wrappers/thrust/execution_policy.cuh"

#include "wrappers/thrust/elementwise_bridge.h"

// complex.h (not .cuh): its device wrappers sit in a device-pass-gated section
// alongside the types -- the vendor-neutral wwrC* ops a cuComplex aggregate
// (which has no operators) needs on the device.
#include "complex.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstddef>
#include <type_traits>

namespace wwr::extension::thrust::device {
namespace {

// A device_ptr wrapping a raw device pointer; Thrust then treats the range as
// device memory and dispatches through the par_on policy rather than copying to
// the host. One spelling for const and non-const T.
template<typename T>
auto dptr(T *p) {
  return ::thrust::device_pointer_cast(p);
}

// ------------------------------------------------------------------------
// Device functors for the named ops. Each is written against the
// vendor-neutral wwr* device primitives so a complex element (cuComplex /
// hipComplex, an operator-less aggregate under CUDA) is handled through wwrC*
// rather than operators it does not have. operator() is plain __device__ and
// const, as the device functor convention requires.
// ------------------------------------------------------------------------

template<typename T>
struct negate_fn {
  __device__ T operator()(const T x) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return make_wwrFloatComplex(-wwrCrealf(x), -wwrCimagf(x));
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return make_wwrDoubleComplex(-wwrCreal(x), -wwrCimag(x));
    } else {
      return -x;
    }
  }
};

template<typename T>
struct abs_fn {
  // Reals only (see UnaryOp::Abs note): a real |x| stays type T. Complex
  // magnitude is a DIFFERENT type (the real scalar), so it is not an
  // elementwise-in-place transform and is deliberately out of scope here.
  __device__ T operator()(const T x) const {
    return x < T{0} ? -x : x;
  }
};

template<typename T>
struct square_fn {
  __device__ T operator()(const T x) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return wwrCmulf(x, x);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return wwrCmul(x, x);
    } else {
      return x * x;
    }
  }
};

template<typename T>
struct plus_fn {
  __device__ T operator()(const T a, const T b) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return wwrCaddf(a, b);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return wwrCadd(a, b);
    } else {
      return a + b;
    }
  }
};

template<typename T>
struct minus_fn {
  __device__ T operator()(const T a, const T b) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return wwrCsubf(a, b);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return wwrCsub(a, b);
    } else {
      return a - b;
    }
  }
};

template<typename T>
struct multiply_fn {
  __device__ T operator()(const T a, const T b) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return wwrCmulf(a, b);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return wwrCmul(a, b);
    } else {
      return a * b;
    }
  }
};

// Equality predicate against a bound value, for the complex replace path:
// cuComplex/hipComplex have no operator==, so thrust::replace (which uses ==)
// cannot run on them. thrust::replace_if with this functor does the same work
// through the component reads.
template<typename T>
struct equals_fn {
  T target_;
  __device__ bool operator()(const T x) const {
    if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      return wwrCrealf(x) == wwrCrealf(target_) && wwrCimagf(x) == wwrCimagf(target_);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      return wwrCreal(x) == wwrCreal(target_) && wwrCimag(x) == wwrCimag(target_);
    } else {
      return x == target_;
    }
  }
};

// Map a UnaryOp selector to its functor type.
template<UnaryOp Op, typename T>
auto unary_functor() {
  if constexpr (Op == UnaryOp::Negate) {
    return negate_fn<T>{};
  } else if constexpr (Op == UnaryOp::Abs) {
    return abs_fn<T>{};
  } else {
    return square_fn<T>{};
  }
}

template<BinaryOp Op, typename T>
auto binary_functor() {
  if constexpr (Op == BinaryOp::Plus) {
    return plus_fn<T>{};
  } else if constexpr (Op == BinaryOp::Minus) {
    return minus_fn<T>{};
  } else {
    return multiply_fn<T>{};
  }
}

} // namespace

template<typename T>
void fill(const wwrStream_t stream, T *out, const std::size_t count, const T value) {
  if (count < 1) {
    return;
  }
  const auto first = dptr(out);
  ::thrust::fill(wwr::par_on(stream), first, first + count, value);
}

template<typename T>
void sequence(const wwrStream_t stream, T *out, const std::size_t count, const T init,
              const T step) {
  if (count < 1) {
    return;
  }
  const auto first = dptr(out);
  ::thrust::sequence(wwr::par_on(stream), first, first + count, init, step);
}

template<typename T>
void replace(const wwrStream_t stream, T *out, const std::size_t count, const T old_value,
             const T new_value) {
  if (count < 1) {
    return;
  }
  const auto first = dptr(out);
  if constexpr (std::is_same_v<T, wwrFloatComplex> || std::is_same_v<T, wwrDoubleComplex>) {
    // No operator== on the complex aggregate -- use replace_if with a
    // component-wise equality predicate (identical work, portable to both).
    ::thrust::replace_if(wwr::par_on(stream), first, first + count, equals_fn<T>{old_value},
                         new_value);
  } else {
    ::thrust::replace(wwr::par_on(stream), first, first + count, old_value, new_value);
  }
}

template<UnaryOp Op, typename T>
void transform_unary(const wwrStream_t stream, const T *in, T *out, const std::size_t count) {
  if (count < 1) {
    return;
  }
  const auto first = dptr(in);
  ::thrust::transform(wwr::par_on(stream), first, first + count, dptr(out),
                      unary_functor<Op, T>());
}

template<BinaryOp Op, typename T>
void transform_binary(const wwrStream_t stream, const T *a, const T *b, T *out,
                       const std::size_t count) {
  if (count < 1) {
    return;
  }
  const auto first = dptr(a);
  ::thrust::transform(wwr::par_on(stream), first, first + count, dptr(b), dptr(out),
                      binary_functor<Op, T>());
}

// ------------------------------------------------------------------------
// Explicit instantiations. One per supported (op x element type), matching the
// extern template lists in elementwise.cppm and instantiations.cpp. The three
// lists cover the same sets and have to stay in step -- a type added here
// without being added there links against nothing.
// ------------------------------------------------------------------------

// fill: every numeric element type, complex included (a plain copy of the value).
#define WWR_INST_FILL(T) template void fill<T>(wwrStream_t, T *, std::size_t, T)
WWR_INST_FILL(float);
WWR_INST_FILL(double);
WWR_INST_FILL(int);
WWR_INST_FILL(unsigned int);
WWR_INST_FILL(long long);
WWR_INST_FILL(unsigned long long);
WWR_INST_FILL(wwrFloatComplex);
WWR_INST_FILL(wwrDoubleComplex);
#undef WWR_INST_FILL

// sequence: ordered-additive types only (reals + integers); complex has no
// meaningful linear step, so it is excluded.
#define WWR_INST_SEQ(T) template void sequence<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_INST_SEQ(float);
WWR_INST_SEQ(double);
WWR_INST_SEQ(int);
WWR_INST_SEQ(unsigned int);
WWR_INST_SEQ(long long);
WWR_INST_SEQ(unsigned long long);
#undef WWR_INST_SEQ

// replace: every numeric element type (reals/ints via ==, complex via the
// component-wise predicate above).
#define WWR_INST_REP(T) template void replace<T>(wwrStream_t, T *, std::size_t, T, T)
WWR_INST_REP(float);
WWR_INST_REP(double);
WWR_INST_REP(int);
WWR_INST_REP(unsigned int);
WWR_INST_REP(long long);
WWR_INST_REP(unsigned long long);
WWR_INST_REP(wwrFloatComplex);
WWR_INST_REP(wwrDoubleComplex);
#undef WWR_INST_REP

// transform unary: Negate and Square over all numeric types; Abs over reals
// only (complex magnitude is a different type, out of scope -- see the enum).
#define WWR_INST_UN(Op, T) template void transform_unary<Op, T>(wwrStream_t, const T *, T *, std::size_t)
WWR_INST_UN(UnaryOp::Negate, float);
WWR_INST_UN(UnaryOp::Negate, double);
WWR_INST_UN(UnaryOp::Negate, int);
WWR_INST_UN(UnaryOp::Negate, long long);
WWR_INST_UN(UnaryOp::Negate, wwrFloatComplex);
WWR_INST_UN(UnaryOp::Negate, wwrDoubleComplex);
WWR_INST_UN(UnaryOp::Square, float);
WWR_INST_UN(UnaryOp::Square, double);
WWR_INST_UN(UnaryOp::Square, int);
WWR_INST_UN(UnaryOp::Square, long long);
WWR_INST_UN(UnaryOp::Square, wwrFloatComplex);
WWR_INST_UN(UnaryOp::Square, wwrDoubleComplex);
WWR_INST_UN(UnaryOp::Abs, float);
WWR_INST_UN(UnaryOp::Abs, double);
WWR_INST_UN(UnaryOp::Abs, int);
WWR_INST_UN(UnaryOp::Abs, long long);
#undef WWR_INST_UN

// transform binary: Plus / Minus / Multiply over all numeric types, complex
// included (via the wwrC* functors above).
#define WWR_INST_BIN(Op, T) \
  template void transform_binary<Op, T>(wwrStream_t, const T *, const T *, T *, std::size_t)
#define WWR_INST_BIN_ALL(Op) \
  WWR_INST_BIN(Op, float);   \
  WWR_INST_BIN(Op, double);  \
  WWR_INST_BIN(Op, int);     \
  WWR_INST_BIN(Op, long long); \
  WWR_INST_BIN(Op, wwrFloatComplex); \
  WWR_INST_BIN(Op, wwrDoubleComplex)
WWR_INST_BIN_ALL(BinaryOp::Plus);
WWR_INST_BIN_ALL(BinaryOp::Minus);
WWR_INST_BIN_ALL(BinaryOp::Multiply);
#undef WWR_INST_BIN_ALL
#undef WWR_INST_BIN

} // namespace wwr::extension::thrust::device
