// random_normal.cu
//
// The device-kernel half of gpumod.extension.random_normal. Shared unchanged
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

namespace gpumod::extension::device {

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
/// `scale_` directly, a complex by complex multiplication (`gpuCmul*`), and a
/// half in single precision, before the narrowing conversion.
///
/// `states_` and `output_` are const, which is what makes the functor
/// non-copy-assignable -- the immutability parallel_for's device_functor
/// concept checks for. `scale_` is deliberately NOT const: a const member of
/// CLASS type (gpuFloatComplex, gpuHalf, ...) is non-trivially-copyable under
/// clang and so fails the same concept's is_trivially_copyable check on a HIP
/// build, while nvcc accepts it -- see parallel_for.cuh's device_functor note.
/// The const pointers already delete copy-assignment, so scale_ need not be
/// const to satisfy the concept. operator() is plain `__device__`.
template<typename T>
struct random_normal_functor {
  gpurandState *const states_;
  T *const output_;
  T scale_;

  __device__ void operator()(const std::size_t i) const {
    if constexpr (std::is_same_v<T, float>) {
      output_[i] = gpurand_normal(&states_[i]) * scale_;
    } else if constexpr (std::is_same_v<T, double>) {
      output_[i] = gpurand_normal_double(&states_[i]) * scale_;
    } else if constexpr (std::is_same_v<T, gpuFloatComplex>) {
      // Each component an independent standard normal, so |z|^2 has variance 2;
      // pass scale = 1/sqrt(2) for a unit-magnitude complex value.
      const float2 value = gpurand_normal2(&states_[i]);
      output_[i] = gpuCmulf(make_gpuFloatComplex(value.x, value.y), scale_);
    } else if constexpr (std::is_same_v<T, gpuDoubleComplex>) {
      const double2 value = gpurand_normal2_double(&states_[i]);
      output_[i] = gpuCmul(make_gpuDoubleComplex(value.x, value.y), scale_);
    } else if constexpr (std::is_same_v<T, gpuHalf>) {
      // Drawn and scaled in single precision, then converted: neither vendor
      // has a native half-precision normal generator.
      output_[i] = gpuFloat2Half(gpurand_normal(&states_[i]) * gpuHalf2Float(scale_));
    } else if constexpr (std::is_same_v<T, gpuBfloat16>) {
      output_[i] = gpuFloat2Bfloat16(gpurand_normal(&states_[i]) * gpuBfloat162Float(scale_));
    } else {
      static_assert(sizeof(T) == 0,
                    "device::random_normal: unhandled output type -- add a branch here "
                    "and an explicit instantiation below");
    }
  }
};

} // namespace

template<typename T>
void random_normal(const gpuStream_t stream, const std::size_t count, gpurandState *states,
                   T *output, const T scale) {
  if (count < 1) {
    return;
  }
  const random_normal_functor<T> functor{states, output, scale};
  parallel_for(stream, count, functor);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's. All three lists cover the same six types.
template void random_normal<float>(gpuStream_t, std::size_t, gpurandState *, float *, float);
template void random_normal<double>(gpuStream_t, std::size_t, gpurandState *, double *, double);
template void random_normal<gpuFloatComplex>(gpuStream_t, std::size_t, gpurandState *,
                                             gpuFloatComplex *, gpuFloatComplex);
template void random_normal<gpuDoubleComplex>(gpuStream_t, std::size_t, gpurandState *,
                                              gpuDoubleComplex *, gpuDoubleComplex);
template void random_normal<gpuHalf>(gpuStream_t, std::size_t, gpurandState *, gpuHalf *, gpuHalf);
template void random_normal<gpuBfloat16>(gpuStream_t, std::size_t, gpurandState *, gpuBfloat16 *,
                                         gpuBfloat16);

} // namespace gpumod::extension::device
