/**
 * @file rand.cuh
 * @brief The cuRAND-or-hipRAND device API, for device-compiled translation units
 *
 * The __device__-qualified generator functions, which no module can export.
 * rand.cppm is the other half: the host API. The generator state types the
 * functions here operate on come from rand.h (included below), the one home
 * rand.cppm, rand.cuh and the extension bridges all draw them from. Together
 * they cover what curand.h + curand_kernel.h cover, split by execution space
 * rather than by header.
 *
 * Link wwr.rand.device -- separate from wwr.device, which a
 * device TU using this almost certainly also wants.
 *
 * Each name is a thin __device__ forwarding template, not a WWR_FUNCTION
 * reference: the generators are an overload set on CUDA and a function
 * template on hipRAND, and a reference can name neither. The state type is
 * deduced, so a call is written once and compiles on both.
 *
 * These are __device__-only, so they can only be called from device code --
 * e.g. a parallel_for functor's __device__ operator().
 *
 * On HIP, wwrrandState and wwrrandStateXORWOW are distinct types; on CUDA they
 * are one. See docs/architecture.md, sections 1 and 4.
 */

#pragma once

// #errors outside a device pass. rand.h carries no such guard of its own (the
// host bridges and rand.cppm include it from host compiles), so rand.cuh adds
// it here for its own __device__ code.
#include "device_guard.h"

// The state types and the vendor kernel headers -- one home, shared with
// rand.cppm and the extension bridges, so a wwrrandState* named in a kernel here
// is the same type the host allocated. See rand.h.
#include "rand.h"

namespace wwr {

// ========================================================================
// Device functions
//
// One forwarding template per name; see this file's header for why these
// cannot be WWR_FUNCTION-style references. The `State` parameter is deduced
// from the argument, so a call reads identically on both backends.
//
// Return types are spelled with the vector types both backends define
// identically (float2, double2), not aliased -- there is nothing to
// translate.
// ========================================================================

/// @brief Initialize one pseudorandom generator state
///
/// Wraps curand_init / hiprand_init -- the 4-argument pseudorandom form. The
/// quasirandom (direction-vector) overloads are not wrapped; see this file's
/// header.
///
/// @param seed Sequence seed; states sharing a seed but differing in
///             `sequence` are independent
/// @param sequence Sequence number for this state -- conventionally the
///                 element index, so each thread draws its own stream
/// @param offset How far into this state's sequence to start
/// @param state [out] The state to initialize
template<typename State>
__device__ __forceinline__ void wwrrand_init(const unsigned long long seed,
                                             const unsigned long long sequence,
                                             const unsigned long long offset, State *state) {
#if defined(WWR_SELECTED_CUDA)
  ::curand_init(seed, sequence, offset, state);
#else
  ::hiprand_init(seed, sequence, offset, state);
#endif
}

/// @brief Draw one float from the standard normal distribution (mean 0, stddev 1)
template<typename State>
__device__ __forceinline__ float wwrrand_normal(State *state) {
#if defined(WWR_SELECTED_CUDA)
  return ::curand_normal(state);
#else
  return ::hiprand_normal(state);
#endif
}

/// @brief Draw two independent standard normal floats at once
///
/// Cheaper than two wwrrand_normal calls: both backends generate normals in
/// pairs (Box-Muller), so the single-value form discards one half.
template<typename State>
__device__ __forceinline__ float2 wwrrand_normal2(State *state) {
#if defined(WWR_SELECTED_CUDA)
  return ::curand_normal2(state);
#else
  return ::hiprand_normal2(state);
#endif
}

/// @brief Draw one double from the standard normal distribution
template<typename State>
__device__ __forceinline__ double wwrrand_normal_double(State *state) {
#if defined(WWR_SELECTED_CUDA)
  return ::curand_normal_double(state);
#else
  return ::hiprand_normal_double(state);
#endif
}

/// @brief Draw two independent standard normal doubles at once
///
/// See wwrrand_normal2 for why the paired form is the cheaper one.
template<typename State>
__device__ __forceinline__ double2 wwrrand_normal2_double(State *state) {
#if defined(WWR_SELECTED_CUDA)
  return ::curand_normal2_double(state);
#else
  return ::hiprand_normal2_double(state);
#endif
}

} // namespace wwr
