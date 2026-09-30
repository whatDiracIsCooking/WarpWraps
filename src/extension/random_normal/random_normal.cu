// random_normal.cu
//
// The device-kernel half of wwr.extension.random_normal. Shared unchanged
// between both backends -- see this directory's CMakeLists.txt for how it is
// compiled as device code under each, why the extension is .cu on both, and why
// there is no per-backend #if in it.
#include "extension/random_normal/random_normal_bridge.h"

#include "bf16.cuh"
#include "complex.cuh"
#include "extension/parallel_for/parallel_for.cuh"
#include "fp16.cuh"
#include "rand.cuh"

#include <cstddef>
#include <type_traits>

namespace wwr::extension::device {

namespace {

/// @brief Draws one standard-normal value per index, scaled by `scale_`
///
/// The dispatch is `if constexpr` on T rather than six functors:
/// each backend's device API offers a different generator per precision, and
/// a float2/double2 pair generator for the complex cases. See
/// this file's interface header for the table. A complex value is drawn as two
/// independent standard normals (no per-component normalization), so its
/// magnitude has variance 2 unless the caller scales it down.
///
/// The drawn value is multiplied by `scale_` before it is stored -- a real by
/// `scale_` directly, a complex by complex multiplication (through the
/// vendor-neutral `wwrCmul*`), and a half in single precision, before the
/// narrowing conversion.
///
/// `states_` and `output_` are const, which is what makes the functor
/// non-copy-assignable -- the immutability parallel_for's device_functor
/// concept checks for. `scale_` is deliberately NOT const: a const member of
/// CLASS type (wwrFloatComplex, wwrHalf, ...) is non-trivially-copyable under
/// clang and so fails the same concept's is_trivially_copyable check on a HIP
/// build, while nvcc accepts it -- see parallel_for.cuh's device_functor note.
/// The const pointers already delete copy-assignment, so scale_ need not be
/// const to satisfy the concept. operator() is plain `__device__`.
template<typename T>
struct random_normal_functor {
  wwrrandState *const states_;
  T *const output_;
  T scale_;

  __device__ void operator()(const std::size_t i) const {
    if constexpr (std::is_same_v<T, float>) {
      output_[i] = wwrrand_normal(&states_[i]) * scale_;
    } else if constexpr (std::is_same_v<T, double>) {
      output_[i] = wwrrand_normal_double(&states_[i]) * scale_;
    } else if constexpr (std::is_same_v<T, wwrFloatComplex>) {
      // Each component an independent standard normal, so |z|^2 has variance 2;
      // pass scale = 1/sqrt(2) for a unit-magnitude complex value. The draw is
      // scaled by a complex multiply through the vendor-neutral wwrC* layer.
      const float2 value = wwrrand_normal2(&states_[i]);
      output_[i] = wwrCmulf(make_wwrFloatComplex(value.x, value.y), scale_);
    } else if constexpr (std::is_same_v<T, wwrDoubleComplex>) {
      const double2 value = wwrrand_normal2_double(&states_[i]);
      output_[i] = wwrCmul(make_wwrDoubleComplex(value.x, value.y), scale_);
    } else if constexpr (std::is_same_v<T, wwrHalf>) {
      // Drawn and scaled in single precision, then converted: neither vendor
      // has a native half-precision normal generator.
      output_[i] = wwrFloat2Half(wwrrand_normal(&states_[i]) * wwrHalf2Float(scale_));
    } else if constexpr (std::is_same_v<T, wwrBfloat16>) {
      output_[i] = wwrFloat2Bfloat16(wwrrand_normal(&states_[i]) * wwrBfloat162Float(scale_));
    } else {
      static_assert(sizeof(T) == 0,
                    "device::random_normal: unhandled output type -- add a branch here "
                    "and an explicit instantiation below");
    }
  }
};

} // namespace

template<typename T>
void random_normal(const wwrStream_t stream, const std::size_t count, wwrrandState *states,
                   T *output, const T scale) {
  if (count < 1) {
    return;
  }
  const random_normal_functor<T> functor{states, output, scale};
  parallel_for(stream, count, functor);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's. All three lists cover the same six types.
template void random_normal<float>(wwrStream_t, std::size_t, wwrrandState *, float *, float);
template void random_normal<double>(wwrStream_t, std::size_t, wwrrandState *, double *, double);
template void random_normal<wwrFloatComplex>(wwrStream_t, std::size_t, wwrrandState *,
                                             wwrFloatComplex *, wwrFloatComplex);
template void random_normal<wwrDoubleComplex>(wwrStream_t, std::size_t, wwrrandState *,
                                              wwrDoubleComplex *, wwrDoubleComplex);
template void random_normal<wwrHalf>(wwrStream_t, std::size_t, wwrrandState *, wwrHalf *, wwrHalf);
template void random_normal<wwrBfloat16>(wwrStream_t, std::size_t, wwrrandState *, wwrBfloat16 *,
                                         wwrBfloat16);

} // namespace wwr::extension::device
