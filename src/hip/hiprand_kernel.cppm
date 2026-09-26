/**
 * @file hiprand_kernel.cppm
 * @brief hipRAND device-state API module wrapper for gpumod project
 *
 * Wraps hiprand/hiprand_kernel.h for the device-side generator STATE TYPES
 * only. CUDA counterpart: the "Device API Types" section of
 * src/cuda/curand.cppm, which wraps curand_kernel.h in the same module as
 * curand.h.
 *
 * A separate module from hiprand.cppm because the two vendor headers have very
 * different weights: hiprand.h is declarations-only, while hiprand_kernel.h
 * pulls in the whole rocRAND device-generator implementation. Host-API
 * consumers keep paying only for hiprand.h.
 *
 * The device FUNCTIONS are absent and cannot be here: they are
 * __device__-qualified, and a module unit is host code. They are reached
 * through src/rand.cuh instead. The state types are plain data, and the
 * host is what allocates and sizes the per-thread array.
 *
 * hiprandState and hiprandStateXORWOW are DISTINCT types here, unlike cuRAND's
 * single aliased type -- docs/architecture.md, section 1, pinned by
 * test/hip/hiprand_kernel.cppm. Also absent: the deprecated hipRandState_t
 * spelling, and MTGP32's host-side parameter types.
 *
 * Usage:
 *   import gpumod.hip.hiprand_kernel;
 *
 *   gpumod::hip::hiprandState state;   // one per thread, allocated on device
 */

module;

// Load-bearing: rocrand_mtgp32.h calls bare printf without declaring it.
// docs/architecture.md, section 7.
#include <cstdio>

#include <hiprand/hiprand_kernel.h>

export module gpumod.hip.hiprand_kernel;

// ========================================================================
// Export hipRAND device state types in gpumod::hip
// (NOT bare gpumod -- see src/hip/README.md "Design decisions")
// ========================================================================

export namespace gpumod::hip {

// ========================================================================
// Device state types -- pseudorandom generators
// ========================================================================
using ::hiprandStateMRG32k3a;
using ::hiprandStateMRG32k3a_t;
using ::hiprandStateMtgp32;
using ::hiprandStateMtgp32_t;
using ::hiprandStatePhilox4_32_10;
using ::hiprandStatePhilox4_32_10_t;
using ::hiprandStateXORWOW;
using ::hiprandStateXORWOW_t;

// ========================================================================
// Device state types -- quasirandom generators
// ========================================================================
using ::hiprandStateScrambledSobol32;
using ::hiprandStateScrambledSobol32_t;
using ::hiprandStateScrambledSobol64;
using ::hiprandStateScrambledSobol64_t;
using ::hiprandStateSobol32;
using ::hiprandStateSobol32_t;
using ::hiprandStateSobol64;
using ::hiprandStateSobol64_t;

// ========================================================================
// Default state type -- its own struct here, not an alias of XORWOW.
// See this file's header.
// ========================================================================
using ::hiprandState;
using ::hiprandState_t;

} // namespace gpumod::hip
