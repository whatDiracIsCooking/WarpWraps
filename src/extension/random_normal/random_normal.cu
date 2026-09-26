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

namespace gpumod::extension::detail {

namespace {

/// @brief Draws one standard-normal value per index, in the output's own type
///
/// The dispatch is `if constexpr` on OutputType rather than six functors:
/// each backend's device API offers a different generator per precision, and
/// a float2/double2 pair generator for the complex cases. See
/// this file's interface header for the table and for why the complex
/// components are scaled.
///
/// Members are const so the functor is not copy-assignable, which is what
/// parallel_for's device_functor concept checks for immutability; operator()
/// is plain `__device__`.
template<typename OutputType>
struct random_normal_functor {
  gpurandState *const states_;
  OutputType *const output_;

  __device__ void operator()(const std::size_t i) const {
    if constexpr (std::is_same_v<OutputType, float>) {
      output_[i] = gpurand_normal(&states_[i]);
    } else if constexpr (std::is_same_v<OutputType, double>) {
      output_[i] = gpurand_normal_double(&states_[i]);
    } else if constexpr (std::is_same_v<OutputType, gpuFloatComplex>) {
      // 1/sqrt(2) per component, so that the COMPLEX VALUE is standard
      // normal (each part variance 1/2, magnitude variance 1) rather
      // than each part separately.
      constexpr float inv_sqrt2 = 0.70710678118654752440f;
      const float2 value = gpurand_normal2(&states_[i]);
      output_[i] = make_gpuFloatComplex(value.x * inv_sqrt2, value.y * inv_sqrt2);
    } else if constexpr (std::is_same_v<OutputType, gpuDoubleComplex>) {
      constexpr double inv_sqrt2 = 0.70710678118654752440;
      const double2 value = gpurand_normal2_double(&states_[i]);
      output_[i] = make_gpuDoubleComplex(value.x * inv_sqrt2, value.y * inv_sqrt2);
    } else if constexpr (std::is_same_v<OutputType, gpuHalf>) {
      // Drawn in single precision and converted: neither vendor has a
      // native half-precision normal generator.
      output_[i] = gpuFloat2Half(gpurand_normal(&states_[i]));
    } else if constexpr (std::is_same_v<OutputType, gpuBfloat16>) {
      output_[i] = gpuFloat2Bfloat16(gpurand_normal(&states_[i]));
    } else {
      static_assert(sizeof(OutputType) == 0,
                    "detail::random_normal: unhandled OutputType -- add a branch here "
                    "and an explicit instantiation below");
    }
  }
};

} // namespace

template<typename OutputType>
void random_normal(const gpuStream_t stream, const std::size_t count, gpurandState *states,
                   OutputType *output) {
  if (count < 1) {
    return;
  }
  const random_normal_functor<OutputType> functor{states, output};
  parallel_for(stream, count, functor);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's. All three lists cover the same six types.
template void random_normal<float>(gpuStream_t, std::size_t, gpurandState *, float *);
template void random_normal<double>(gpuStream_t, std::size_t, gpurandState *, double *);
template void random_normal<gpuFloatComplex>(gpuStream_t, std::size_t, gpurandState *,
                                             gpuFloatComplex *);
template void random_normal<gpuDoubleComplex>(gpuStream_t, std::size_t, gpurandState *,
                                              gpuDoubleComplex *);
template void random_normal<gpuHalf>(gpuStream_t, std::size_t, gpurandState *, gpuHalf *);
template void random_normal<gpuBfloat16>(gpuStream_t, std::size_t, gpurandState *, gpuBfloat16 *);

} // namespace gpumod::extension::detail
