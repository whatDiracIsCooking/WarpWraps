// init_state.cu
//
// The device-kernel half of wwr.extension.init_state. Shared unchanged
// between both backends: under CUDA the .cu extension is all CMake needs, under
// HIP this directory's CMakeLists.txt forces LANGUAGE CXX back on and links
// hip::device PRIVATE so clang compiles it with `-x hip`. The extension matches
// the .cuh device headers this includes -- in this project .cu/.cuh means
// "device-compiled, whichever backend", not "CUDA only".
//
// There is no per-backend #if here: rand.h and runtime.cuh resolve
// every backend difference this TU would otherwise have to spell twice, so
// the functor and the launcher are written once. rand.h is #included (not a
// .cuh) because it carries the rand layer's device generators in a device-pass
// gated section as well as the state types -- see src/rand.h.
#include "extension/init_state/init_state_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "rand.h"

#include <cstddef>

namespace wwr::extension::device {

namespace {

/// @brief Initializes one generator state per index
///
/// Members are const so the functor is not copy-assignable, which is what
/// parallel_for's device_functor concept checks for immutability; operator()
/// is plain `__device__` -- it runs only inside parallel_for's kernel, and its
/// callability is constrained on that kernel template, not on device_functor.
struct init_state_functor {
  wwrrandState *const states_;
  const unsigned long long seed_;
  const unsigned long long sequence_offset_;
  const unsigned long long offset_;

  __device__ void operator()(const std::size_t i) const {
    // sequence_offset_ + i, not a constant: states on DIFFERENT
    // subsequences of one seed are independent, states on the same one
    // are identical.
    wwrrand_init(seed_, sequence_offset_ + i, offset_, &states_[i]);
  }
};

} // namespace

void init_state(const wwrStream_t stream, const std::size_t count, wwrrandState *states,
                const unsigned long long seed, const unsigned long long sequence_offset,
                const unsigned long long offset) {
  if (count < 1) {
    return;
  }
  const init_state_functor functor{states, seed, sequence_offset, offset};
  parallel_for(stream, count, functor);
}

} // namespace wwr::extension::device
