/**
 * @file rand.h
 * @brief The single cuRAND / hipRAND vendor-include point for the rand layer
 *
 * A src/-root header wrapping the whole rand vendor surface, split by compile
 * context rather than by file. It carries three things:
 *   - the device generator STATE TYPES (always) -- the vendor types that cross
 *     the host/device boundary by pointer, aliased under wwr* names once;
 *   - the vendor HOST declarations (always #included), which rand.cppm binds its
 *     wwr* host names to (see below);
 *   - the __device__ generator functions (only in a device pass -- the gated
 *     section at the bottom).
 * So rand.cppm re-exports the state types and binds the host API, a device .cu
 * #includes this header to call the generators in a kernel, and the extension
 * bridges (init_state_bridge.h, random_normal_bridge.h) name wwrrandState* in a
 * global module fragment where an `import` cannot reach. It reaches
 * selected_backend.h directly, not device_guard.h, so it carries no device-pass
 * #error and compiles in a host TU too -- the device section gates itself out
 * there. This is the `.h` + `.cppm` shape for a vendor header with BOTH host and
 * device symbols; a purely device-only wrapper stays a `.cuh` with
 * device_guard.h's #error (atomic, cooperative_groups, wmma, parallel_for).
 *
 * It also #includes the vendor HOST header (curand.h / hiprand.h) beside the
 * kernel one, so that it is the one place the vendor headers enter the rand
 * layer. rand.cppm draws its whole host API from here -- binding wwr* references
 * and aliases straight to the ::curand* / ::hiprand* declarations with the _RAW
 * macros -- and so imports no raw vendor module. That works because the host API
 * is a real external-linkage library: a reference or type alias needs only the
 * declaration this header supplies. A module wrapping static-inline vendor math
 * (complex, fp16, bf16) could NOT do this and must keep importing its raw module
 * for that module's external-linkage wrappers. See backend.h and
 * docs/architecture.md, section 12. The raw modules (wwr.cuda.curand,
 * wwr.hip.hiprand, wwr.hip.hiprand_kernel) remain as single-vendor host-API
 * surfaces, off the neutral layer's path.
 *
 * It differs from complex.h / runtime.h in one way that is the whole design
 * trade: their vendor header is host-cheap (cuComplex.h, cuda_runtime_api.h);
 * the state types live only in curand_kernel.h / hiprand_kernel.h, which are
 * heavy (hiprand_kernel.h drags in the whole rocRAND device-generator
 * implementation). The host header added beside it is cheap by comparison.
 * Including them here means the parse lands in every TU that #includes rand.h --
 * but that is exactly three host TUs: rand.cppm's own compile (which builds the
 * wwr.rand BMI) and the two extension wrapper module units, whose global module
 * fragments must #include a bridge because a GMF cannot import. Every consumer
 * that `import`s wwr.rand or the extension modules pays nothing: the BMI exports
 * names, never the header. That bounded cost buys one source of truth -- for the
 * state-type list (in place of the copies the old rand.cuh, rand.cppm and the
 * forward-declaring bridge each carried) and for the host API (in place of the
 * import). See docs/architecture.md, section 1, and src/README.md.
 *
 * wwrrandState and wwrrandStateXORWOW are one type on CUDA and two on HIP; the
 * _t forms and the other states are all named against the real vendor typedefs
 * here, which is why this header includes the vendor kernel headers rather than
 * forward-declaring (a typedef cannot be forward-declared).
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is included from host TUs and must not #error.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

// Host API (curand.h) and device generator state types (curand_kernel.h). Both,
// because this header is the single vendor-include point for the whole rand
// layer: rand.cppm draws the host API from here, a device .cu the __device__
// generators (gated section at the bottom), and both the state types. The host
// header is cheap; the kernel header is the heavy one this arrangement is
// careful about (see below).
#include <curand.h>
#include <curand_kernel.h>

#else

// <cstdio> first, and it is load-bearing: hiprand_kernel.h ->
// hiprand_kernel_rocm.h -> rocrand/rocrand_kernel.h -> rocrand/rocrand_mtgp32.h
// calls bare `printf` without declaring it. A real `-x hip` compile usually
// drags a declaration in through hip_runtime.h before this point, so the
// failure only shows up when this header is reached first -- include it here so
// the order of includes in the consuming TU cannot matter. Same reason
// src/hip/hiprand_kernel.cppm does it. See docs/architecture.md, section 7.
#include <cstdio>

// Host API (hiprand.h) and device state types (hiprand_kernel.h) -- the split
// cuRAND does not have; here they are two headers, and this file includes both
// so it is the one vendor-include point for the rand layer. See above.
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#endif

namespace wwr {

#if defined(WWR_SELECTED_CUDA)

// Pseudorandom generators
using wwrrandStateXORWOW = ::curandStateXORWOW;
using wwrrandStateXORWOW_t = ::curandStateXORWOW_t;
using wwrrandStateMRG32k3a = ::curandStateMRG32k3a;
using wwrrandStateMRG32k3a_t = ::curandStateMRG32k3a_t;
using wwrrandStateMtgp32 = ::curandStateMtgp32;
using wwrrandStateMtgp32_t = ::curandStateMtgp32_t;
using wwrrandStatePhilox4_32_10 = ::curandStatePhilox4_32_10;
using wwrrandStatePhilox4_32_10_t = ::curandStatePhilox4_32_10_t;

// Quasirandom generators
using wwrrandStateSobol32 = ::curandStateSobol32;
using wwrrandStateSobol32_t = ::curandStateSobol32_t;
using wwrrandStateScrambledSobol32 = ::curandStateScrambledSobol32;
using wwrrandStateScrambledSobol32_t = ::curandStateScrambledSobol32_t;
using wwrrandStateSobol64 = ::curandStateSobol64;
using wwrrandStateSobol64_t = ::curandStateSobol64_t;
using wwrrandStateScrambledSobol64 = ::curandStateScrambledSobol64;
using wwrrandStateScrambledSobol64_t = ::curandStateScrambledSobol64_t;

// Default state -- IS wwrrandStateXORWOW here, unlike HIP
using wwrrandState = ::curandState;
using wwrrandState_t = ::curandState_t;

#else

// Pseudorandom generators
using wwrrandStateXORWOW = ::hiprandStateXORWOW;
using wwrrandStateXORWOW_t = ::hiprandStateXORWOW_t;
using wwrrandStateMRG32k3a = ::hiprandStateMRG32k3a;
using wwrrandStateMRG32k3a_t = ::hiprandStateMRG32k3a_t;
using wwrrandStateMtgp32 = ::hiprandStateMtgp32;
using wwrrandStateMtgp32_t = ::hiprandStateMtgp32_t;
using wwrrandStatePhilox4_32_10 = ::hiprandStatePhilox4_32_10;
using wwrrandStatePhilox4_32_10_t = ::hiprandStatePhilox4_32_10_t;

// Quasirandom generators
using wwrrandStateSobol32 = ::hiprandStateSobol32;
using wwrrandStateSobol32_t = ::hiprandStateSobol32_t;
using wwrrandStateScrambledSobol32 = ::hiprandStateScrambledSobol32;
using wwrrandStateScrambledSobol32_t = ::hiprandStateScrambledSobol32_t;
using wwrrandStateSobol64 = ::hiprandStateSobol64;
using wwrrandStateSobol64_t = ::hiprandStateSobol64_t;
using wwrrandStateScrambledSobol64 = ::hiprandStateScrambledSobol64;
using wwrrandStateScrambledSobol64_t = ::hiprandStateScrambledSobol64_t;

// Default state -- its own struct here, NOT wwrrandStateXORWOW
using wwrrandState = ::hiprandState;
using wwrrandState_t = ::hiprandState_t;

#endif

} // namespace wwr

// ========================================================================
// Device generator functions -- present only in a device-compile pass
//
// The __device__-qualified generators, gated behind the compiler's own
// device-pass macros so the rest of this header still compiles in a host TU
// (where __device__ is not a keyword, so these must simply be ABSENT rather
// than #error). A .cu that #includes rand.h gets them; rand.cppm and the
// extension bridges, compiled as host C++, do not -- they use only the types
// above and (rand.cppm) the host declarations. This is the device half that
// once lived in the separate rand.cuh, folded in so a device consumer includes
// one neutral header. Link wwr.rand.device for the include path and the RNG
// library.
//
// One forwarding template per name, not a WWR_FUNCTION reference: the
// generators are an overload set on CUDA and a function template on hipRAND,
// and a reference can name neither. The State type is deduced, so a call reads
// identically on both backends. Return types use the vector types both backends
// spell identically (float2, double2) -- nothing to translate.
//
// These are __device__-only, callable only from device code -- e.g. a
// parallel_for functor's __device__ operator(). See docs/architecture.md,
// sections 1 and 4.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

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

#endif // device-compile pass
