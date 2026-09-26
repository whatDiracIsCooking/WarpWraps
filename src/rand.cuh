/**
 * @file rand.cuh
 * @brief The cuRAND-or-hipRAND device API, for device-compiled translation units
 *
 * The __device__-qualified generator functions, which no module can export.
 * rand.cppm is the other half: the host API and the generator state types.
 * Together they cover what curand.h + curand_kernel.h cover, split by
 * execution space rather than by header.
 *
 * Link gpumod.rand.device -- separate from gpumod.device, which a
 * device TU using this almost certainly also wants.
 *
 * Each name is a thin __device__ forwarding template, not a GPUMOD_FUNCTION
 * reference: the generators are an overload set on CUDA and a function
 * template on hipRAND, and a reference can name neither. The state type is
 * deduced, so a call is written once and compiles on both.
 *
 * These are __device__-only, so they can only be called from device code --
 * e.g. a parallel_for functor's __device__ operator().
 *
 * On HIP, gpurandState and gpurandStateXORWOW are distinct types; on CUDA they
 * are one. See docs/architecture.md, sections 1 and 4.
 */

#pragma once

// GPUMOD_SELECTED_CUDA / GPUMOD_SELECTED_HIP, and #errors outside a device pass;
// these vendor headers are device-only.
#include "device_guard.h"

#if defined(GPUMOD_SELECTED_CUDA)

#include <curand_kernel.h>

#else

// <cstdio> first, and it is load-bearing: hiprand_kernel.h ->
// hiprand_kernel_rocm.h -> rocrand/rocrand_kernel.h -> rocrand/rocrand_mtgp32.h
// calls bare `printf` without declaring it. A real `-x hip` compile usually
// drags a declaration in through hip_runtime.h before this point, so the
// failure only shows up when this header is reached first -- include it here
// so the order of includes in the consuming TU cannot matter. Same reason
// src/hip/hiprand_kernel.cppm does it.
#include <cstdio>

#include <hiprand/hiprand_kernel.h>

#endif

namespace gpumod {

// ========================================================================
// Generator state types
//
// The same types rand.cppm exports under these names, spelled without
// modules. A gpurandState* allocated host-side and a gpurandState* named in
// a kernel here are the same type, so the kernel signature mangles to match.
// ========================================================================

#if defined(GPUMOD_SELECTED_CUDA)

// Pseudorandom generators
using gpurandStateXORWOW = ::curandStateXORWOW;
using gpurandStateXORWOW_t = ::curandStateXORWOW_t;
using gpurandStateMRG32k3a = ::curandStateMRG32k3a;
using gpurandStateMRG32k3a_t = ::curandStateMRG32k3a_t;
using gpurandStateMtgp32 = ::curandStateMtgp32;
using gpurandStateMtgp32_t = ::curandStateMtgp32_t;
using gpurandStatePhilox4_32_10 = ::curandStatePhilox4_32_10;
using gpurandStatePhilox4_32_10_t = ::curandStatePhilox4_32_10_t;

// Quasirandom generators
using gpurandStateSobol32 = ::curandStateSobol32;
using gpurandStateSobol32_t = ::curandStateSobol32_t;
using gpurandStateScrambledSobol32 = ::curandStateScrambledSobol32;
using gpurandStateScrambledSobol32_t = ::curandStateScrambledSobol32_t;
using gpurandStateSobol64 = ::curandStateSobol64;
using gpurandStateSobol64_t = ::curandStateSobol64_t;
using gpurandStateScrambledSobol64 = ::curandStateScrambledSobol64;
using gpurandStateScrambledSobol64_t = ::curandStateScrambledSobol64_t;

// Default state -- IS gpurandStateXORWOW here, unlike HIP
using gpurandState = ::curandState;
using gpurandState_t = ::curandState_t;

#else

// Pseudorandom generators
using gpurandStateXORWOW = ::hiprandStateXORWOW;
using gpurandStateXORWOW_t = ::hiprandStateXORWOW_t;
using gpurandStateMRG32k3a = ::hiprandStateMRG32k3a;
using gpurandStateMRG32k3a_t = ::hiprandStateMRG32k3a_t;
using gpurandStateMtgp32 = ::hiprandStateMtgp32;
using gpurandStateMtgp32_t = ::hiprandStateMtgp32_t;
using gpurandStatePhilox4_32_10 = ::hiprandStatePhilox4_32_10;
using gpurandStatePhilox4_32_10_t = ::hiprandStatePhilox4_32_10_t;

// Quasirandom generators
using gpurandStateSobol32 = ::hiprandStateSobol32;
using gpurandStateSobol32_t = ::hiprandStateSobol32_t;
using gpurandStateScrambledSobol32 = ::hiprandStateScrambledSobol32;
using gpurandStateScrambledSobol32_t = ::hiprandStateScrambledSobol32_t;
using gpurandStateSobol64 = ::hiprandStateSobol64;
using gpurandStateSobol64_t = ::hiprandStateSobol64_t;
using gpurandStateScrambledSobol64 = ::hiprandStateScrambledSobol64;
using gpurandStateScrambledSobol64_t = ::hiprandStateScrambledSobol64_t;

// Default state -- its own struct here, NOT gpurandStateXORWOW
using gpurandState = ::hiprandState;
using gpurandState_t = ::hiprandState_t;

#endif

// ========================================================================
// Device functions
//
// One forwarding template per name; see this file's header for why these
// cannot be GPUMOD_FUNCTION-style references. The `State` parameter is deduced
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
__device__ __forceinline__ void gpurand_init(const unsigned long long seed,
                                             const unsigned long long sequence,
                                             const unsigned long long offset, State *state) {
#if defined(GPUMOD_SELECTED_CUDA)
  ::curand_init(seed, sequence, offset, state);
#else
  ::hiprand_init(seed, sequence, offset, state);
#endif
}

/// @brief Draw one float from the standard normal distribution (mean 0, stddev 1)
template<typename State>
__device__ __forceinline__ float gpurand_normal(State *state) {
#if defined(GPUMOD_SELECTED_CUDA)
  return ::curand_normal(state);
#else
  return ::hiprand_normal(state);
#endif
}

/// @brief Draw two independent standard normal floats at once
///
/// Cheaper than two gpurand_normal calls: both backends generate normals in
/// pairs (Box-Muller), so the single-value form discards one half.
template<typename State>
__device__ __forceinline__ float2 gpurand_normal2(State *state) {
#if defined(GPUMOD_SELECTED_CUDA)
  return ::curand_normal2(state);
#else
  return ::hiprand_normal2(state);
#endif
}

/// @brief Draw one double from the standard normal distribution
template<typename State>
__device__ __forceinline__ double gpurand_normal_double(State *state) {
#if defined(GPUMOD_SELECTED_CUDA)
  return ::curand_normal_double(state);
#else
  return ::hiprand_normal_double(state);
#endif
}

/// @brief Draw two independent standard normal doubles at once
///
/// See gpurand_normal2 for why the paired form is the cheaper one.
template<typename State>
__device__ __forceinline__ double2 gpurand_normal2_double(State *state) {
#if defined(GPUMOD_SELECTED_CUDA)
  return ::curand_normal2_double(state);
#else
  return ::hiprand_normal2_double(state);
#endif
}

} // namespace gpumod
