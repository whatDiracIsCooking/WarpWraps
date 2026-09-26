/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.random_normal
 *
 * Parallel generation of standard-normal values into a typed device array, one
 * value per element, using an array of states already initialized by
 * gpumod.extension.init_state. Draws from the standard normal distribution
 * (mean 0, standard deviation 1), optionally multiplied by a `scale` argument
 * that defaults to 1. The kernel lives in random_normal.cu, device-compiled
 * (see this directory's CMakeLists.txt).
 *
 * The wrapper is an unconstrained template: the supported output types are
 * exactly the ones explicitly instantiated below, and any other type fails to
 * link rather than being rejected by a concept. Those types, and how each is
 * drawn:
 *
 * | Output type      | Drawn with                                        |
 * |------------------|---------------------------------------------------|
 * | float            | gpurand_normal                                    |
 * | double           | gpurand_normal_double                             |
 * | gpuFloatComplex  | gpurand_normal2                                   |
 * | gpuDoubleComplex | gpurand_normal2_double                            |
 * | gpuHalf          | gpurand_normal, converted                         |
 * | gpuBfloat16      | gpurand_normal, converted                         |
 *
 * Each complex component is drawn as an independent standard normal, so a
 * complex value has component variance 1 and magnitude variance 2 -- it is not
 * standard normal as a whole. Pass `scale = 1/sqrt(2)` if you want the complex
 * value's magnitude to have unit variance. The half types are drawn in single
 * precision and converted -- neither vendor has a native half-precision normal
 * generator.
 *
 * Usage:
 *   import gpumod.extension.init_state;
 *   import gpumod.extension.random_normal;
 *   using namespace gpumod::extension;
 *
 *   auto device = std::make_shared<DeviceHandle>();
 *   auto stream = device->alloc_stream().get();
 *   DeviceBuffer<gpurandState> states(n, device);
 *   DeviceBuffer<float> values(n, device);
 *   init_state(stream, n, states.data(), seed);
 *   random_normal(stream, n, states.data(), values.data());
 *
 * This and gpumod.extension.init_state were split out of the single
 * gpumod.extension.rand module (two partitions), itself the port of the
 * CUDA-only gpumod.extension.curand.* (removed).
 */

module;

#include "extension/random_normal/random_normal_bridge.h"

export module gpumod.extension.random_normal;

import std;
import gpumod.runtime_api;
import gpumod.rand;
import gpumod.complex;
import gpumod.fp16;
import gpumod.bf16;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace -- the same
// shape the CUDA-only original used.
namespace gpumod::extension {

/**
 * @brief Draw `count` standard-normal values into `output`
 *
 * Consumes one state per element: `states[i]` advances as `output[i]` is
 * drawn, so the same array fed to a second call continues the streams rather
 * than repeating them. `states` must have been initialized by
 * gpumod.extension.init_state's init_state.
 *
 * Returns immediately without launching anything when `count` is 0.
 *
 * @tparam T One of the explicitly instantiated types below; see this
 *                    file's header for the list and how each is drawn
 * @param stream Stream to launch on; the work is asynchronous
 * @param count Number of values to draw
 * @param states Device array of at least `count` initialized states
 * @param output Device array of at least `count` elements
 * @param scale Each drawn value is multiplied by this before being stored;
 *              defaults to 1 (unscaled standard normal). For a real type this
 *              scales the standard deviation; for a complex type it multiplies
 *              each value by the complex `scale`.
 */
export template<typename T>
void random_normal(const gpuStream_t stream, const std::size_t count, gpurandState *states,
                   T *output, const T scale = T{1.0f}) {
  // device:: is load-bearing -- without it this names itself.
  device::random_normal(stream, count, states, output, scale);
}

// Instantiated once in instantiations.cpp, not at every call site. The
// definitions these resolve to are in random_normal.cu, compiled as device
// code -- see random_normal_bridge.h for why that boundary is untyped.
extern template void random_normal<float>(gpuStream_t, std::size_t, gpurandState *, float *, float);
extern template void random_normal<double>(gpuStream_t, std::size_t, gpurandState *, double *,
                                           double);
extern template void random_normal<gpuFloatComplex>(gpuStream_t, std::size_t, gpurandState *,
                                                    gpuFloatComplex *, gpuFloatComplex);
extern template void random_normal<gpuDoubleComplex>(gpuStream_t, std::size_t, gpurandState *,
                                                     gpuDoubleComplex *, gpuDoubleComplex);
extern template void random_normal<gpuHalf>(gpuStream_t, std::size_t, gpurandState *, gpuHalf *,
                                            gpuHalf);
extern template void random_normal<gpuBfloat16>(gpuStream_t, std::size_t, gpurandState *,
                                                gpuBfloat16 *, gpuBfloat16);

} // namespace gpumod::extension
