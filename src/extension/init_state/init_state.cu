// init_state.cu
//
// The device-kernel half of gpumod.extension.init_state. Shared unchanged
// between both backends: under CUDA the .cu extension is all CMake needs, under
// HIP this directory's CMakeLists.txt forces LANGUAGE CXX back on and links
// hip::device PRIVATE so clang compiles it with `-x hip`. The extension matches
// the .cuh device headers this includes -- in this project .cu/.cuh means
// "device-compiled, whichever backend", not "CUDA only".
//
// There is no per-backend #if here: rand.cuh and runtime.cuh resolve
// every backend difference this TU would otherwise have to spell twice, so
// the functor and the launcher are written once.
#include "extension/init_state/init_state_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "rand.cuh"

#include <cstddef>

namespace gpumod::extension::detail {

namespace {

/// @brief Initializes one generator state per index
///
/// Members are const so the functor is not copy-assignable, which is what
/// parallel_for's device_functor concept checks for immutability; operator()
/// is plain `__device__` -- it runs only inside parallel_for's kernel, and its
/// callability is constrained on that kernel template, not on device_functor.
struct init_state_functor {
  gpurandState *const states_;
  const unsigned long long seed_;
  const unsigned long long sequence_offset_;
  const unsigned long long offset_;

  __device__ void operator()(const std::size_t i) const {
    // sequence_offset_ + i, not a constant: states on DIFFERENT
    // subsequences of one seed are independent, states on the same one
    // are identical.
    gpurand_init(seed_, sequence_offset_ + i, offset_, &states_[i]);
  }
};

} // namespace

void init_state(const gpuStream_t stream, const std::size_t count, gpurandState *states,
                const unsigned long long seed, const unsigned long long sequence_offset,
                const unsigned long long offset) {
  if (count < 1) {
    return;
  }
  const init_state_functor functor{states, seed, sequence_offset, offset};
  parallel_for(stream, count, functor);
}

} // namespace gpumod::extension::detail
