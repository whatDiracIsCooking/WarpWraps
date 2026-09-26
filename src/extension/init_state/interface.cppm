/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.init_state
 *
 * Parallel initialization of an array of cuRAND / hipRAND DEVICE-API generator
 * states: one state per element, each seeded onto its own subsequence, so that
 * a later kernel can draw an independent stream of numbers per thread without
 * any coordination. Pairs with gpumod.extension.random_normal, which draws
 * from the states this module initializes.
 *
 * The work is dispatched through gpumod.extension.parallel_for; the kernel
 * itself lives in init_state.cu, device-compiled (see this directory's
 * CMakeLists.txt).
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
 *
 *   init_state(stream, n, states.data(), seed);
 *   random_normal(stream, n, states.data(), values.data());
 *
 * The state array is the caller's: it is what makes a second call continue the
 * same streams instead of repeating them, and what lets one kernel's states be
 * reused by another. gpurandState comes from gpumod.rand.
 *
 * This and gpumod.extension.random_normal were split out of the single
 * gpumod.extension.rand module (two partitions), itself the port of the
 * CUDA-only gpumod.extension.curand.* (removed).
 */

module;

#include "extension/init_state/init_state_bridge.h"

export module gpumod.extension.init_state;

import std;
import gpumod.runtime_api;
import gpumod.rand;

export namespace gpumod::extension {

/**
 * @brief Initialize an array of generator states in parallel, one per element
 *
 * Each state `i` is initialized onto subsequence `sequence_offset + i` of the
 * same seed, which is what makes the per-element streams independent. Seeding
 * every state identically instead (same seed, same subsequence) would make
 * every thread draw the SAME numbers.
 *
 * Returns immediately without launching anything when `count` is 0.
 *
 * @param stream Stream to launch on; the work is asynchronous
 * @param count Number of states to initialize
 * @param states Device array of at least `count` states
 * @param seed Generator seed -- states differing only in subsequence are
 *             independent draws from the same seed
 * @param sequence_offset Subsequence number of the first state; state `i`
 *                        gets `sequence_offset + i`. Use it to carve further
 *                        independent blocks out of one seed.
 * @param offset How far into each state's own subsequence to skip ahead
 */
void init_state(const gpuStream_t stream, const std::size_t count, gpurandState *states,
                const unsigned long long seed = 0, const unsigned long long sequence_offset = 0,
                const unsigned long long offset = 0) {
  // detail:: is load-bearing -- without it this names itself.
  detail::init_state(stream, count, states, seed, sequence_offset, offset);
}

} // namespace gpumod::extension
